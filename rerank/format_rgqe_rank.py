import torch
import argparse
from collections import defaultdict
import os
import json


def load_predictions(input_path):
	pred_list = []
	for file_name in os.listdir(input_path):
		if file_name.endswith('.pt'):
			preds = torch.load(os.path.join(input_path, file_name))
			pred_list.extend(preds)
	answer_q_scores = defaultdict(lambda: defaultdict(float))
	for prediction in pred_list:
		answer_id = prediction['id']
		question_id = prediction['question_id']
		# score = prediction['pos_score']
		score = prediction['pos_score'] - prediction['neg_score']
		answer_q_scores[answer_id][question_id] = score
	return answer_q_scores


def save_predictions(question_scores, output_path, run_name):
	with open(output_path, 'w') as f:
		for question_id, question_scores in question_scores.items():
			for idx, (doc_pass_id, score) in enumerate(question_scores):
				rank = idx + 1
				f.write(f'{question_id}\tQ0\t{doc_pass_id}\t{rank}\t{score:.8f}\t{run_name}\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	args = parser.parse_args()

	input_path = args.input_path
	output_path = args.output_path
	question_scores = load_predictions(input_path)
	with open(output_path, 'w') as f:
		json.dump(question_scores, f, indent=2)
