import torch
import argparse
from collections import defaultdict
import os
from transformers import T5Tokenizer
from tqdm import tqdm
from multiprocessing import Pool
import json
import re


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', required=True)
parser.add_argument('-o', '--output_path', required=True)
parser.add_argument('-t', '--tokenizer_name', default='t5-base')
parser.add_argument('-ps', '--num_processes', default=16, type=int)
args = parser.parse_args()

model_path = args.model_path
output_path = args.output_path
tokenizer_name = args.tokenizer_name
num_processes = args.num_processes
tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
p = re.compile(r'\bc\w+d\w*\b')
replace_set = {
	'copvid',
	'copid',
	'cod',
	'cid',
	'covidos',
	'covidia',
	'comcid',
	'comid',
	'civid',
	'cid',
	'covenid',
	'corvid',
	'cardiovid',
	'covizid',
	'coincid',
	'covied',
	'covvid',
	'combid'
}
replace_text = 'covid'
replace_length = len(replace_text)

def encode_prediction(prediction):
	answer_id = prediction['id']
	# [sample_size, max_seq_len]
	sample_seq_ids = prediction['samples']
	samples = []
	seen_text = set()
	for sample_ids in sample_seq_ids:
		sample_txt = tokenizer.decode(
			sample_ids.tolist(),
			skip_special_tokens=True
		)
		text = sample_txt
		modified_offset = 0
		for m in p.finditer(sample_txt):
			start_idx, end_idx = m.span()
			m_text = m.group()
			if m_text in replace_set:
				start_idx += modified_offset
				end_idx += modified_offset
				text = text[:start_idx] + replace_text + text[end_idx:]
				modified_offset += replace_length - len(m_text)

		if text not in seen_text:
			samples.append(text)
			seen_text.add(text)
	return answer_id, samples


def load_predictions(model_path):
	pred_list = []
	for file_name in os.listdir(model_path):
		if file_name.endswith('.pt'):
			preds = torch.load(os.path.join(model_path, file_name))
			pred_list.extend(preds)
	answer_queries = defaultdict(list)
	with Pool(processes=num_processes) as p:
		for answer_id, samples in tqdm(
				p.imap_unordered(encode_prediction, pred_list), total=len(pred_list)):
			answer_queries[answer_id] = samples

	return answer_queries


def save_predictions(answer_queries, output_path):
	# with open(output_path, 'w') as f:
	# 	for answer_id, answer_qs in answer_queries.items():
	# 		aq_text = '\t'.join(answer_qs)
	# 		f.write(f'{answer_id}\t{aq_text}\n')
	with open(output_path, 'w') as f:
		json.dump(answer_queries, f, indent=2)


answer_queries = load_predictions(model_path)
save_predictions(answer_queries, output_path)

