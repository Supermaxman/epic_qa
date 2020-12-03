from collections import OrderedDict, defaultdict
import argparse





class Segment:
    def __init__(self, context, start, end, score):
        self.context = context
        self.start = start
        self.end = end
        self.score = score

    @property
    def length(self):
        return self.end - self.start + 1

    def __le__(self, other):
        if self.context < other.context:
            return True
        elif self.context == other.context:
            if self.start < other.start:
                return True
            elif self.start == other.start:
                return self.end < other.end
        return False

    def __gt__(self, other):
        return not self < other

    def __str__(self):
        return f'{self.context}:{self.start}-{self.end}\t{self.score}'


class Context:
    def __init__(self, name, scores, ngram_length):
        self.name = name
        self.scores = scores
        self.max_scores = scores.copy()
        self.pruned = [False] * len(scores)
        self.ngram_length = ngram_length

    @property
    def num_scores(self):
        return len(self.scores)

    def __str__(self):
        return f'{self.name}: {self.scores}'


#
# Structure: [max_ngram_length][num_contexts]Context
#
class ScoreIndex:
    def __init__(self, ngram_contexts):
        self.ngram_contexts = ngram_contexts

    @property
    def max_ngram_length(self):
        return len(self.ngram_contexts)

    @property
    def num_contexts(self):
        return len(self.ngram_contexts[0])

    def num_scores_at(self, context_index):
        return len(self.ngram_contexts[0][context_index].scores)

    def score_at(self, context_index, start, end):
        return self.ngram_contexts[end - start][context_index].scores[start]

    def max_score_at(self, context_index, start, end):
        return self.ngram_contexts[end - start][context_index].max_scores[start]

    def set_max_score(self, context_index, start, end, value):
        self.ngram_contexts[end - start][context_index].max_scores[start] = value

    def pruned_at(self, context_index, start, end):
        return self.ngram_contexts[end - start][context_index].pruned[start]

    def set_pruned(self, context_index, start, end, value):
        self.ngram_contexts[end - start][context_index].pruned[start] = value

    #
    # Precondition: Segments must be sorted
    # Return Type: [max_gram_length][]Context
    #
    @staticmethod
    def group_by_ngram_and_context(segments, max_gram_length):
        # Group by n-gram
        ngram_segments = []
        for i in range(1, max_gram_length + 1):
            ngram_segments.append([])
            for segment in segments:
                if segment.length == i:
                    ngram_segments[-1].append(segment)

        # Group by context
        ngram_contexts = []
        for index, segment_group in enumerate(ngram_segments):
            context_dict = OrderedDict()
            for segment in segments:
                if segment.context not in context_dict:
                    context_dict[segment.context] = Context(segment.context, [], index + 1)
            for segment in segment_group:
                context_dict[segment.context].scores.append(segment.score)
                context_dict[segment.context].max_scores.append(segment.score)
                context_dict[segment.context].pruned.append(False)
            ngram_contexts.append(list(context_dict.values()))

        return ScoreIndex(ngram_contexts)


class RerankResult:
    def __init__(self, top_segments, num_segments_processed):
        self.top_segments = top_segments
        self.num_segments_processed = num_segments_processed

    def __str__(self):
        value = ''
        for segment in self.top_segments:
            value += f'{str(segment)}\n'
        return value


def sort_scores(score_index: ScoreIndex, limit=None):
    segments = []

    for contexts in score_index.ngram_contexts:
        for context in contexts:
            for i in range(context.num_scores):
                if not context.pruned[i]:
                    segments.append(Segment(
                        context=context.name,
                        start=i,
                        end=i + context.ngram_length - 1,
                        score=context.scores[i]
                    ))
    segments.sort(key=lambda x: x.score)
    segments.reverse()
    if limit:
        segments = segments[:limit]
    return segments


def rerank_naive(score_index: ScoreIndex, limit):
    sorted_scores = sort_scores(score_index, limit)
    num_segments_processed = 0
    for contexts in score_index.ngram_contexts:
        for context in contexts:
            num_segments_processed += sum([1 for p in context.pruned if not p])
    return RerankResult(sorted_scores, num_segments_processed)


def rerank_pruned(score_index: ScoreIndex, limit, t):
    num_segments_processed = 0

    # Calculate and prune uni-grams
    for context_index in range(score_index.num_contexts):
        for start in range(score_index.num_scores_at(context_index)):
            if score_index.score_at(context_index, start, start) < t:
                score_index.set_pruned(context_index, start, start, True)
            num_segments_processed += 1

    #
    # Calculate and prune n-grams
    # In this loop, we will assume that we cannot calculate future n-grams.
    # In other words, we can only refer to n-grams of ngram_index - n, where n >= 0.
    #
    for ngram_index in range(1, score_index.max_ngram_length):
        for context_index in range(score_index.num_contexts):
            for start in range(score_index.num_scores_at(context_index) - ngram_index):
                # Skip if all sub-segments are pruned.
                # Since we memoize prune flags, we only need to get two values.
                pruned_1 = score_index.pruned_at(context_index, start, start + ngram_index - 1)
                pruned_2 = score_index.pruned_at(context_index, start + 1, start + ngram_index)
                if pruned_1 and pruned_2:
                    try:
                        score_index.set_pruned(context_index, start, start + ngram_index, True)
                    except Exception as e:
                        print(f'{context_index}, {start}, {start + ngram_index}')
                        print(f'  - {context_index}, {start}, {start + ngram_index - 1}')
                        print(f'  - {context_index}, {start+1}, {start + ngram_index}')
                        raise e
                    continue

                # Include segment in the batch and calculate score.
                ngram_score = score_index.score_at(context_index, start, start + ngram_index)
                num_segments_processed += 1

                # Prune if score(segment) < max(scores(sub_segments)).
                # Since we memoize maximum scores, we only need to get two values.
                score_1 = score_index.max_score_at(context_index, start, start + ngram_index - 1)
                score_2 = score_index.max_score_at(context_index, start + 1, start + ngram_index)
                max_sub_segment_score = max(score_1, score_2)

                max_score = max(ngram_score, max_sub_segment_score)
                score_index.set_max_score(context_index, start, start + ngram_index, max_score)

                if ngram_score < max_sub_segment_score:
                    score_index.set_pruned(context_index, start, start + ngram_index, True)

    sorted_scores = sort_scores(score_index, limit)
    return RerankResult(sorted_scores, num_segments_processed)


def rerank_prune(segments, t, n):
    score_index = ScoreIndex.group_by_ngram_and_context(segments, n)
    # naive_result = rerank_naive(score_index, limit=10)
    pruned_result = rerank_pruned(score_index, limit=1000, t=t)

    return pruned_result


def read_segments(filename):
    query_segments = {}
    with open(filename) as file:
        for line in file:
            columns = line.split()
            segment_query = columns[0]
            location = columns[2].split('-')
            if segment_query not in query_segments:
                query_segments[segment_query] = []
            query_segments[segment_query].append(
                Segment(
                    context=f'{location[0]}-{location[1]}',
                    start=int(location[2]),
                    end=int(location[3]),
                    score=float(columns[4])
                )
            )
    return query_segments


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred_path', required=True)
    parser.add_argument('-o', '--output_path', required=True)
    parser.add_argument('-n', '--top_n_gram', default=3, type=int)
    parser.add_argument('-t', '--threshold', default=0.0, type=float)
    args = parser.parse_args()

    pred_path = args.pred_path
    output_path = args.output_path
    output_name = output_path.split('/')[-1].replace('.pred', '')

    rerank_scores = read_segments(pred_path)
    print(list(rerank_scores.keys()))
    with open(output_path, 'w') as f:
        for question_id, query_segments in rerank_scores.items():
            query_segments.sort(key=compare_segment)
            print(f'{question_id}: {len(query_segments)}')

            result = rerank_prune(query_segments, t=args.threshold, n=args.top_n_gram)
            for idx, seg in enumerate(result.top_segments):
                rank = idx + 1
                f.write(
                    f'{question_id}\t'
                    f'Q0\t'
                    f'{seg.context}-{seg.start}-{seg.end}\t'
                    f'{rank}\t'
                    f'{seg.score}\t'
                    f'{output_name}\n')
