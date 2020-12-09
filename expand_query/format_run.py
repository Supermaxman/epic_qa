
from collections import defaultdict
import argparse
import json


def sort_results(rerank_scores, query_answer_samples):
	results = defaultdict(list)
	for question_id, q_scores in rerank_scores.items():
		q_results = []
		for answer_id, answer_score in q_scores:
			try:
				query_samples = query_answer_samples[answer_id]
			except KeyError:
				query_samples = []
			answer_results = {
				'answer_id': answer_id,
				'score': answer_score,
				'samples': query_samples
			}
			q_results.append(answer_results)

		q_results = list(sorted(q_results, key=lambda x: x['score'], reverse=True))
		results[question_id] = q_results
	return results


def read_run(scores_path):
	rerank_scores = defaultdict(list)
	with open(scores_path) as f:
		for line in f:
			line = line.strip()
			if line:
				question_id, _, answer_id, rank, score, _ = line.split('\t')
				score = float(score)
				rerank_scores[question_id].append((answer_id, score))
	return rerank_scores


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-s', '--scores_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)

	args = parser.parse_args()

	input_path = args.input_path
	scores_path = args.scores_path
	output_path = args.output_path

	with open(input_path) as f:
		query_answer_samples = json.load(f)

	rerank_scores = read_run(scores_path)
	rerank_results = sort_results(rerank_scores, query_answer_samples)

	with open(output_path, 'w') as f:
		json.dump(rerank_results, f, indent=2)

