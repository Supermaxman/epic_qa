from collections import defaultdict
import argparse


def write_results(rerank_scores, output_path, output_name, threshold):
	with open(output_path, 'w') as f:
		for question_id, q_scores in rerank_scores.items():
			new_q_scores = []
			for answer_id, answer_score, query_samples in q_scores:
				nugget_count = 0.0
				unique_samples = set()
				for sample_text, entail_prob in query_samples:
					if sample_text in unique_samples:
						continue
					unique_samples.add(sample_text)
					if entail_prob > threshold:
						nugget_count += 1
				sample_score = nugget_count / len(unique_samples)
				# full_score = (1.0 + sample_score) * answer_score
				full_score = sample_score
				new_q_scores.append((answer_id, answer_score, sample_score, full_score))

			new_q_scores = list(sorted(new_q_scores, key=lambda x: x[-1], reverse=True))
			rank = 1
			for answer_id, answer_score, sample_score, full_score in new_q_scores:
				f.write(f'{question_id}\tQ0\t{answer_id}\t{rank}\t{full_score:.8f}\t{output_name}\n')
				rank += 1


def read_run(answer_query_path, expand_path, scores_path):
	answer_queries = {}
	with open(answer_query_path) as f:
		for line in f:
			line = line.strip()
			if line:
				line_list = line.split('\t')
				answer_id = line_list[0]
				queries = line_list[1:]
				answer_queries[answer_id] = queries

	query_answer_sample_probs = defaultdict(list)
	with open(expand_path) as f:
		for line in f:
			line = line.strip()
			if line:
				question_id, answer_id, sample_id, entail_prob = line.split('\t')
				sample_id = int(sample_id)
				entail_prob = float(entail_prob)
				sample_text = answer_queries[answer_id][sample_id]
				query_answer_sample_probs[(question_id, answer_id)].append((sample_text, entail_prob))

	rerank_scores = defaultdict(list)
	with open(scores_path) as f:
		for line in f:
			line = line.strip()
			if line:
				question_id, _, answer_id, rank, score, _ = line.split('\t')
				score = float(score)
				query_samples = query_answer_sample_probs[(question_id, answer_id)]
				rerank_scores[question_id].append((answer_id, score, query_samples))
	return rerank_scores


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-a', '--answer_query_path', required=True)
	parser.add_argument('-e', '--expand_path', required=True)
	parser.add_argument('-s', '--scores_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-t', '--threshold', default=0.5, type=float)

	args = parser.parse_args()
	# 'runs/consumer/pruned_biobert_msmarco_multi_sentence'
	answer_query_path = args.answer_query_path
	expand_path = args.expand_path
	scores_path = args.scores_path
	output_path = args.output_path
	output_name = output_path.split('/')[-1].replace('.txt', '').replace('.pred', '')

	threshold = args.threshold

	rerank_scores = read_run(answer_query_path, expand_path, scores_path)
	write_results(rerank_scores, output_path, output_name, threshold)



