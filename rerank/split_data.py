
import torch
import argparse
from collections import defaultdict
import os
import json


def load_predictions(model_path):
	pred_list = []
	for file_name in os.listdir(model_path):
		if file_name.endswith('.pt'):
			preds = torch.load(os.path.join(model_path, file_name))
			pred_list.extend(preds)
	question_scores = defaultdict(list)
	# # {query_id}\tQ0\t{doc_pass_id}\t{rank}\t{score:.4f}\t{run_name}
	for prediction in pred_list:
		doc_pass_id = prediction['id']
		question_id = prediction['question_id']
		# score = prediction['pos_score']
		score = prediction['pos_score'] - prediction['neg_score']
		question_scores[question_id].append((doc_pass_id, score))

	sorted_scores = {}
	for question_id, q_scores in question_scores.items():
		sorted_scores[question_id] = list(sorted(q_scores, key=lambda x: x[1], reverse=True))

	return sorted_scores


def save_predictions(question_scores, output_path, run_name):
	with open(output_path, 'w') as f:
		for question_id, question_scores in question_scores.items():
			for idx, (doc_pass_id, score) in enumerate(question_scores):
				rank = idx + 1
				f.write(f'{question_id}\tQ0\t{doc_pass_id}\t{rank}\t{score:.8f}\t{run_name}\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-l', '--label_path', required=True)
	parser.add_argument('-s', '--search_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-ds', '--dataset', required=True)
	parser.add_argument('-sr', '--split_ratio', default=0.8, type=float)
	args = parser.parse_args()

	label_path = args.label_path
	search_path = args.search_path
	output_path = args.output_path
	split_ratio = args.split_ratio
	dataset_prefix = 'C' if args.dataset.lower() == 'consumer' else 'E'

	if not os.path.exists(output_path):
		os.mkdir(output_path)

	labels = []
	with open(label_path, 'r') as f:
		all_labels = json.load(f)
		for label in all_labels:
			if label['question_id'].startswith(dataset_prefix):
				labels.append(label)

	num_labels = len(labels)
	train_idx = int(split_ratio * num_labels)
	train_labels = labels[:train_idx]
	val_labels = labels[train_idx:]
	print(f'#train_qs={len(train_labels)}, #val_qs={len(val_labels)}')
	with open(os.path.join(output_path, 'train.json'), 'w') as f:
		json.dump(train_labels, f, indent=2)
	with open(os.path.join(output_path, 'val.json'), 'w') as f:
		json.dump(val_labels, f, indent=2)
