
class Segment:
    def __init__(self, context, start, end, score):
        self.context = context
        self.start = start
        self.end = end
        self.score = score

    def __lt__(self, other):
        if self.context < other.context:
            return True
        elif self.context == other.context:
            if self.start < other.start:
                return True
            elif self.start == other.start:
                return self.end < other.end
        return False

    def __str__(self):
        return f'{self.context},{self.start},{self.end},{self.score}\n'


def read_segments(filename, query):
    segments = []
    with open(filename) as file:
        for line in file:
            columns = line.split()
            segment_query = int(columns[0])
            location = columns[2].split('-')
            if segment_query == query:
                segments.append(Segment(
                    context=f'{location[0]}-{location[1]}',
                    start=int(location[2]),
                    end=int(location[3]),
                    score=float(columns[4])
                ))
    return segments


def sort_single_query(query):
    segments = read_segments('consumer_biobert_msmarco_multi_sentence', query)
    segments.sort()

    with open(f'sorted_scores_{query}.csv', 'w') as file:
        for segment in segments:
            file.write(str(segment))


def sort_all_queries():
    for i in range(42):
        print(f'Sorting Q{i + 1}...')

        segments = read_segments('consumer_biobert_msmarco_multi_sentence', i + 1)
        segments.sort()

        with open(f'sorted_scores_{i + 1}.csv', 'w') as file:
            for segment in segments:
                file.write(str(segment))


sort_all_queries()