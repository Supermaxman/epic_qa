import json
from tqdm import tqdm
import argparse
from collections import defaultdict
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from model_utils import QuestionAnsweringBert


def find_ngrams(input_list, n):
	return zip(*[input_list[i:] for i in range(n)])


class PassageDataset(Dataset):
	def __init__(self, root_dir, file_names, query, multi_sentence, n_gram_max):
		self.root_dir = root_dir
		self.file_names = file_names
		self.examples = []
		self.query = query
		self.multi_sentence = multi_sentence
		self.n_gram_max = n_gram_max
		doc_lookup = {}
		for d_id, p_id in self.file_names:
			if d_id not in doc_lookup:
				doc_lookup[d_id] = []
			doc_lookup[d_id].append(p_id)

		for d_id, p_ids in doc_lookup.items():
			file_path = os.path.join(self.root_dir, d_id + '.json')
			with open(file_path) as f:
				doc = json.load(f)
			for p_id in p_ids:
				passage = doc['contexts'][p_id]
				context_examples = []
				for s_id, sentence in enumerate(passage['sentences']):
					example = {
						'id': f'{d_id}-{p_id}-{s_id}-{s_id}',
						'd_id': d_id,
						'p_id': p_id,
						's_id': s_id,
						'text': passage['text'][sentence['start']:sentence['end']],
						'query': query
					}
					self.examples.append(example)
					context_examples.append(example)
				if self.multi_sentence:
					# generate sentence n-grams from 2 to n_gram_max of contiguous sentence spans
					for k in range(2, self.n_gram_max+1):
						for ex_list in find_ngrams(context_examples, n=k):
							ex_first = ex_list[0]
							ex_last = ex_list[-1]
							example = {
								'id': f'{ex_first["d_id"]}-{ex_first["p_id"]}-{ex_first["s_id"]}-{ex_last["s_id"]}',
								'd_id': ex_first['d_id'],
								'p_id': ex_first['p_id'],
								's_id': f'{ex_first["s_id"]}-{ex_last["s_id"]}',
								'text': ' '.join([ex['text'] for ex in ex_list]),
								'query': ex_first['query']
							}
							self.examples.append(example)

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		example = self.examples[idx]

		return example


parser = argparse.ArgumentParser()
parser.add_argument('-q', '--query_path', required=True)
parser.add_argument('-c', '--collection_path', required=True)
parser.add_argument('-l', '--search_run', required=True)
parser.add_argument('-r', '--run_path', required=True)
parser.add_argument('-rm', '--rerank_model', default='nboost/pt-biobert-base-msmarco')
parser.add_argument('-bs', '--batch_size', default=64, type=int)
parser.add_argument('-ml', '--max_length', default=512, type=int)
parser.add_argument('-ms', '--multi_sentence', default=False, action='store_true')
parser.add_argument('-ng', '--n_gram_max', default=3, type=int)
parser.add_argument('-cm', '--custom_model', default=False, action='store_true')
parser.add_argument('-db', '--debug', default=False, action='store_true')
parser.add_argument('-t', '--threshold', default=0.0, type=float)
parser.add_argument('-lp', '--label_path', default='data/prelim_judgments.json')

args = parser.parse_args()

# 'expert'
collection_path = args.collection_path
search_run = args.search_run
# 'baseline_doc'
run_name = args.run_path
# expert_questions_prelim.json
query_path = args.query_path
run_path = args.run_path
debug = args.debug
threshold = args.threshold
label_path = args.label_path

rerank_model_name = args.rerank_model
tokenizer_name = args.rerank_model
batch_size = args.batch_size
max_length = args.max_length
multi_sentence = args.multi_sentence
n_gram_max = args.n_gram_max
custom_model = args.custom_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

print(f'Loading tokenizer: {tokenizer_name}')
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

print(f'Loading model: {rerank_model_name}')
if custom_model:
	model = torch.load(os.path.join(rerank_model_name, 'pytorch_model.bin'))
else:
	model = AutoModelForSequenceClassification.from_pretrained(rerank_model_name)
model.to(device)
model.eval()


def collate_batch(examples):
	passages = [x['text'] for x in examples]
	queries = [x['query'] for x in examples]
	pairs = list(zip(queries, passages))
	batch = tokenizer.batch_encode_plus(
		# query, passage
		batch_text_or_text_pairs=pairs,
		add_special_tokens=True,
		padding=True,
		return_tensors='pt',
		truncation='only_second',
		max_length=max_length
	)
	batch['id'] = [x['id'] for x in examples]
	batch['d_id'] = [x['d_id'] for x in examples]
	batch['p_id'] = [x['p_id'] for x in examples]
	batch['s_id'] = [x['s_id'] for x in examples]
	return batch


with open(query_path) as f:
	queries = json.load(f)

qrels = defaultdict(list)
with open(search_run) as f:
	for line in f:
		line = line.strip().split()
		if line:
			query_id, _, doc_id, _, _, _ = line
			query_id = int(query_id)
			doc_id, pass_id = doc_id.split('-')
			pass_id = int(pass_id)
			qrels[query_id].append((doc_id, pass_id))

keep_ids = None
if label_path:
	keep_ids = set()
	with open(label_path) as f:
		questions = json.load(f)
		for question in questions:
			question_id = question['question_id']
			keep_ids.add(question_id)

print(f'Running reranking on passages and writing results to {run_path}...')
pass_scores = defaultdict(list)
for query_id, query in tqdm(enumerate(queries, start=1), total=len(queries)):
	if debug and query_id != 1:
		continue
	question_id = query['question_id']
	if keep_ids is not None and question_id not in keep_ids:
		continue
	query_labels = qrels[query_id]
	assert len(query_labels) > 0
	dataset = PassageDataset(
		collection_path,
		query_labels,
		query['question'],
		multi_sentence,
		n_gram_max
	)
	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=1,
		collate_fn=collate_batch
	)

	with torch.no_grad():
		# score_layer = torch.nn.Softmax(dim=-1)
		for batch in tqdm(dataloader, total=len(dataloader)):
			logits = model(
				input_ids=batch['input_ids'].to(device),
				token_type_ids=batch['token_type_ids'].to(device),
				attention_mask=batch['attention_mask'].to(device)
			)[0]

			if custom_model:
				scores = (-logits).cpu().numpy()
			else:
				scores = logits[:, 1].cpu().numpy()

			# make larger score mean better answer
			pass_scores[query_id].extend(zip(batch['id'], scores))

	# sort large to small
	pass_scores[query_id] = list(sorted(pass_scores[query_id], key=lambda x: x[1], reverse=True))


print(f'Saving results...')
with open(run_path, 'w') as fo:
	for query_id, passages in pass_scores.items():
		for rank, (doc_pass_sent_id, score) in enumerate(passages, start=1):
			if score > threshold:
				line = f'{query_id}\tQ0\t{doc_pass_sent_id}\t{rank}\t{score:.8f}\t{run_name}\n'
				fo.write(line)
print('Done!')
