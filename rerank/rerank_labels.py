import json
from tqdm import tqdm
import argparse
from collections import defaultdict
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader


class PassageDataset(Dataset):
	def __init__(self, root_dir, file_names, query):
		self.root_dir = root_dir
		self.file_names = file_names
		self.examples = []
		self.query = query
		for file_name in self.file_names:
			file_path = os.path.join(self.root_dir, file_name + '.json')
			with open(file_path) as f:
				doc = json.load(f)
				for p_id, passage in enumerate(doc['contexts']):
					example = {
						'id': f'{file_name}-{p_id}',
						'file_name': file_name,
						'p_id': p_id,
						'text': passage['text'],
						'query': query
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
parser.add_argument('-l', '--label_path', required=True)
parser.add_argument('-r', '--run_path', required=True)
parser.add_argument('-rm', '--rerank_model', default='nboost/pt-biobert-base-msmarco')
parser.add_argument('-bs', '--batch_size', default=16, type=int)

# TODO consider other reranking models

args = parser.parse_args()

# 'expert'
collection_path = args.collection_path
label_path = args.label_path
# 'baseline_doc'
run_name = args.run_path
# expert_questions_prelim.json
query_path = args.query_path
run_path = args.run_path

rerank_model_name = args.rerank_model
batch_size = args.batch_size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

print(f'Loading tokenizer: {rerank_model_name}')
tokenizer = AutoTokenizer.from_pretrained(rerank_model_name)

print(f'Loading model: {rerank_model_name}')
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
		max_length=512
	)
	batch['id'] = [x['id'] for x in examples]
	batch['file_name'] = [x['file_name'] for x in examples]
	batch['p_id'] = [x['p_id'] for x in examples]
	return batch


with open(query_path) as f:
	queries = json.load(f)


qrels = defaultdict(list)
with open(label_path, 'r') as f:
	for line in f:
		line = line.strip().split()
		if line:
			query_id, _, doc_id, dq_rank = line
			query_id = int(query_id)
			dq_rank = int(dq_rank)
			if dq_rank > 0:
				qrels[query_id].append(doc_id)


print(f'Running reranking on passages and writing results to {run_path}...')
pass_scores = defaultdict(list)
for query_id, query in tqdm(enumerate(queries, start=1), total=len(queries)):
	query_labels = qrels[query_id]
	assert len(query_labels) > 0
	dataset = PassageDataset(collection_path, query_labels, query['question'])
	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=1,
		collate_fn=collate_batch
	)
	for batch in tqdm(dataloader, total=len(dataloader)):
		scores = model(
			input_ids=batch['input_ids'].to(device),
			token_type_ids=batch['token_type_ids'].to(device),
			attention_mask=batch['attention_mask'].to(device)
		)[0][:, 0].data.cpu().numpy()
		# make larger score mean better answer
		pass_scores[query_id].extend(zip(batch['id'], [-x for x in scores]))

	# sort large to small
	pass_scores[query_id] = list(sorted(pass_scores[query_id], key=lambda x: x[1], reverse=True))


print(f'Saving results...')
with open(run_path, 'w') as fo:
	for query_id, passages in pass_scores.items():
		for rank, (doc_pass_id, score) in enumerate(passages, start=1):
			line = f'{query_id}\tQ0\t{doc_pass_id}\t{rank}\t{score:.4f}\t{run_name}\n'
			fo.write(line)
print('Done!')
