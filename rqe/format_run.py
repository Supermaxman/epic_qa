
from collections import defaultdict
import argparse
import json


def sort_results(rerank_scores, query_answer_samples):
	results = defaultdict(list)
	for question_id, q_scores in rerank_scores.items():
		q_results = []
		for answer_id, answer_score in q_scores:
			answer_questions = set()
			try:
				query_samples = query_answer_samples[question_id][answer_id]
			except KeyError:
				query_samples = []
			filtered_samples = []
			for q_sample in query_samples:
				# entail_prob = q_sample['entail_prob']
				sample_text = q_sample['sample_text']
				if sample_text in answer_questions:
					continue
				answer_questions.add(sample_text)
				filtered_samples.append(q_sample)
			answer_results = {
				'answer_id': answer_id,
				'score': answer_score,
				'samples': filtered_samples
			}
			q_results.append(answer_results)

		q_results = list(sorted(q_results, key=lambda x: x['score'], reverse=True))
		sample_count = 0
		kept_count = 0
		for result in q_results[:100]:
			num_samples = len(result['samples'])
			sample_count += num_samples
			if num_samples > 0:
				kept_count += 1
		print(f'top-100 %kept={kept_count/100:.2f} (#num_samples={sample_count})')
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
	parser.add_argument('-r', '--rqe_path', required=True)
	parser.add_argument('-s', '--scores_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)

	args = parser.parse_args()

	rqe_path = args.rqe_path
	scores_path = args.scores_path
	output_path = args.output_path

	with open(rqe_path) as f:
		query_answer_samples = json.load(f)

	rerank_scores = read_run(scores_path)
	rerank_results = sort_results(rerank_scores, query_answer_samples)

	with open(output_path, 'w') as f:
		json.dump(rerank_results, f, indent=2)

