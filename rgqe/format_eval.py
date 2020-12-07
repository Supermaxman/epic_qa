
from collections import defaultdict
import argparse
import json


def create_results(query_results, sample_entail_pairs):
	lower_ranked_entailed_answers = sample_entail_pairs
	higher_ranked_entailed_answers = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
	for question_id, answer_a_entailed_answers in sample_entail_pairs.items():
		for answer_a_id, answer_b_entailed_answers in answer_a_entailed_answers.items():
			for answer_b_id, a_b_sample_pairs in answer_b_entailed_answers.items():
				for sample_a, sample_b in a_b_sample_pairs:
					higher_ranked_entailed_answers[question_id][answer_b_id][answer_a_id].append((sample_b, sample_a))

	results = defaultdict(list)
	for question_id, q_scores in query_results.items():
		q_results = []
		for answer in q_scores:
			answer_id = answer['answer_id']
			# list of answers and samples are ranked lower than current answer
			answer_lower_entailed = lower_ranked_entailed_answers[question_id][answer_id]
			# list of answers and samples are ranked higher than current answer
			answer_higher_entailed = higher_ranked_entailed_answers[question_id][answer_id]
			# query_samples = answer['samples']
			# TODO more in-depth entailment checks
			# for q_sample in query_samples:
			# 	sample_id = q_sample['sample_id']
			# num answers with at least one sampled question which entails a current answer question
			num_higher_entailed_answers = len(answer_higher_entailed)
			num_higher_entailed_samples = 0
			for higher_answer, sample_pairs in answer_higher_entailed.items():
				num_higher_entailed_samples += len(sample_pairs)
			num_lower_entailed_answers = len(answer_lower_entailed)
			num_lower_entailed_samples = 0
			for lower_answer, sample_pairs in answer_lower_entailed.items():
				num_lower_entailed_samples += len(sample_pairs)

			# if an answer which is ranked higher has at least one overlapping entailed sample question
			# then we assume duplicate information is provided by the lower-ranked answer.
			# TODO work on these
			if num_higher_entailed_answers > 0:
				continue
			q_results.append(answer)

		results[question_id] = q_results
	return results


def write_run(results, output_path, output_name):
	with open(output_path, 'w') as f:
		for question_id, q_results in results.items():
			rank = 1
			for answer in q_results:
				answer_id = answer['answer_id']
				answer_score = answer['score']
				f.write(f'{question_id}\tQ0\t{answer_id}\t{rank}\t{answer_score:.8f}\t{output_name}\n')
				rank += 1


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--results_path', required=True)
	parser.add_argument('-g', '--rgqe_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)

	args = parser.parse_args()

	# TODO figure out
	results_path = args.results_path
	rgqe_path = args.rgqe_path
	output_path = args.output_path
	output_name = output_path.split('/')[-1].replace('.txt', '').replace('.pred', '')

	with open(results_path) as f:
		query_results = json.load(f)

	with open(rgqe_path) as f:
		sample_entail_pairs = json.load(f)

	rerank_results = create_results(query_results, sample_entail_pairs)

	with open(output_path) as f:
		query_answer_samples = json.load(f)

	write_run(rerank_results, output_path, output_name)





