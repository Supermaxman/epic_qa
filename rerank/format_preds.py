import torch
import argparse
from collections import defaultdict


def load_predictions(predictions_path):
	pred_list = torch.load(predictions_path)
	question_scores = defaultdict(list)
	# # {query_id}\tQ0\t{doc_pass_id}\t{rank}\t{score:.4f}\t{run_name}
	for prediction in pred_list:
		doc_pass_id = prediction['id']
		question_id = prediction['question_id']
		score = prediction['pos_score']
		question_scores[question_id].append((doc_pass_id, score))

	sorted_scores = {}
	for question_id, question_scores in question_scores.items():
		sorted_scores[question_id] = list(sorted(question_scores, key=lambda x: x[1], reverse=True))

	return sorted_scores


def save_predictions(question_scores, output_path, run_name):
	with open(output_path, 'w') as f:
		for question_id, question_scores in question_scores.items():
			for idx, (doc_pass_id, score) in enumerate(question_scores):
				rank = idx + 1
				f.write(f'{question_id}\tQ0\t{doc_pass_id}\t{rank}\t{score:.8f}\t{run_name}\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--run_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	args = parser.parse_args()

	run_path = args.run_path
	output_path = args.output_path
	output_name = output_path.split('/')[-1].replace('.txt', '')

	question_scores = load_predictions(run_path)
	save_predictions(question_scores, output_path, output_name)
