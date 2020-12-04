import torch
import argparse
from collections import defaultdict
import os
from transformers import T5Tokenizer
from tqdm import tqdm


def load_predictions(model_path, tokenizer):
	pred_list = []
	for file_name in os.listdir(model_path):
		if file_name.endswith('.pt'):
			preds = torch.load(os.path.join(model_path, file_name))
			pred_list.extend(preds)
	answer_queries = defaultdict(list)
	# # {query_id}\tQ0\t{doc_pass_id}\t{rank}\t{score:.4f}\t{run_name}
	for prediction in tqdm(pred_list):
		answer_id = prediction['id']
		# [sample_size, max_seq_len]
		sample_seq_ids = prediction['samples']
		samples = []
		for sample_ids in sample_seq_ids:
			sample_txt = tokenizer.decode(sample_ids.tolist(), skip_special_tokens=True)
			samples.append(sample_txt)
		answer_queries[answer_id] = samples

	return answer_queries


def save_predictions(answer_queries, output_path):
	with open(output_path, 'w') as f:
		for answer_id, answer_queries in answer_queries.items():
			aq_text = '\t'.join(answer_queries)
			f.write(f'{answer_id}\t{aq_text}\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--model_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-t', '--tokenizer_name', default='t5-base')
	args = parser.parse_args()

	model_path = args.model_path
	output_path = args.output_path
	tokenizer_name = args.tokenizer_name
	tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)

	answer_queries = load_predictions(model_path, tokenizer)
	save_predictions(answer_queries, output_path)
