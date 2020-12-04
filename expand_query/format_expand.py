import torch
import argparse
from collections import defaultdict
import os


def load_predictions(model_path, tokenizer):
	pred_list = []
	for file_name in os.listdir(model_path):
		if file_name.endswith('.pt'):
			preds = torch.load(os.path.join(model_path, file_name))
			pred_list.extend(preds)
	answer_queries = defaultdict(list)
	# # {query_id}\tQ0\t{doc_pass_id}\t{rank}\t{score:.4f}\t{run_name}
	for prediction in pred_list:
		answer_id = prediction['id']
		# [sample_size, max_seq_len]
		sample_seq_ids = prediction['samples']
		samples = []
		for sample_ids in sample_seq_ids:
			sample_txt = tokenizer.decode(sample_ids, skip_special_tokens=True)
			samples.append(sample_txt)
		answer_queries[answer_id] = samples

	for answer_id, answer_queries in answer_queries.items():
		pass

	return sorted_scores


def save_predictions(question_scores, output_path, run_name):
	with open(output_path, 'w') as f:
		for question_id, question_scores in question_scores.items():
			for idx, (doc_pass_id, score) in enumerate(question_scores):
				rank = idx + 1
				f.write(f'{question_id}\tQ0\t{doc_pass_id}\t{rank}\t{score:.8f}\t{run_name}\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--model_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	args = parser.parse_args()

	model_path = args.model_path
	output_path = args.output_path
	output_name = output_path.split('/')[-1].replace('.txt', '').replace('.pred', '')

	question_scores = load_predictions(model_path)
	save_predictions(question_scores, output_path, output_name)
