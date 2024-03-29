import json
from tqdm import tqdm
import argparse
from collections import defaultdict
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument('-dt', '--doc_type', required=True)
parser.add_argument('-q', '--query', required=True)
parser.add_argument('-cp', '--collection_path', required=True)
parser.add_argument('-ri', '--input_run_name', required=True)
parser.add_argument('-r', '--run_name', required=True)

# TODO consider other reranking models
rerank_model_name = 'nboost/pt-biobert-base-msmarco'
batch_size = 16

args = parser.parse_args()

# 'expert'
doc_type = args.doc_type
collection_path = args.collection_path
# 'baseline_doc'
run_name = args.run_name
input_run_name = args.input_run_name
# expert_questions_prelim.json
query_path = f'data/{doc_type}/{args.query}'
run_path = f'runs/{doc_type}/{run_name}'
input_run_path = f'runs/{doc_type}/{input_run_name}'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

tokenizer = AutoTokenizer.from_pretrained(rerank_model_name)

model = AutoModelForSequenceClassification.from_pretrained(rerank_model_name)
model.to(device)
model.eval()


def extract_passage(doc_id, pass_id):
	with open(os.path.join(collection_path, doc_id + '.json'), 'r') as f:
		doc = json.load(f)
	# passages = []
	context_text = doc['contexts'][pass_id]['text']
	# context_passages = []
	# for sentence in context['sentences']:
	# 	sentence_text = context_text[sentence['start']:sentence['end']]
	# 	if sentence_text.strip():
	# 		context_passages.append(sentence_text)
	# additional_passages = []
	# TODO add back in when efficient
	# if len(context_passages) >= 2:
	# 	bi_grams = list([' '.join(x) for x in zip(context_passages[:1], context_passages[1:])])
	# 	additional_passages.extend(bi_grams)
	# if len(context_passages) >= 3:
	# 	tri_grams = list([' '.join(x) for x in zip(context_passages[:2], context_passages[1:1], context_passages[2:])])
	# 	additional_passages.extend(tri_grams)
	# context_passages = context_passages + additional_passages
	# passages.extend(context_passages)
	# passages.append(context_text)
	return context_text


def rerank(query, passages):
	pairs = [(query, passage) for passage in passages]
	all_scores = []
	num_batches = int(np.ceil(len(pairs) / batch_size))
	for b_idx in tqdm(range(num_batches), total=num_batches):
		batch = tokenizer.batch_encode_plus(
			# query, passage
			batch_text_or_text_pairs=pairs[b_idx * batch_size:(b_idx + 1) * batch_size],
			add_special_tokens=True,
			padding=True,
			return_tensors='pt',
			truncation='only_second',
			max_length=512
		)

		scores = model(
			input_ids=batch['input_ids'].to(device),
			token_type_ids=batch['token_type_ids'].to(device),
			attention_mask=batch['attention_mask'].to(device)
		)[0][:, 0].data.cpu().numpy()
		all_scores.extend(scores)

	passages_sorted = list(sorted(zip(all_scores, range(len(passages))), key=lambda x: x[0]))
	return passages_sorted


query_hits = defaultdict(list)
with open(input_run_path) as f:
	for line in f:
		line = line.strip().split()
		if line:
			query_id, _, doc_id, _, _, _ = line
			query_id = int(query_id)
			doc_id, pass_id = doc_id.split('-')
			pass_id = int(pass_id)
			query_hits[query_id].append((doc_id, pass_id))

with open(query_path) as f:
	queries = json.load(f)

print(f'Running reranking on {input_run_name} and writing results to {run_path}...')
with open(run_path, 'w') as fo:
	for query_id, query in tqdm(enumerate(queries, start=1), total=len(queries)):
		hits = query_hits[query_id]

		all_passages = []
		doc_lookup = {}
		for doc_id, pass_id in hits:
			passage = extract_passage(doc_id, pass_id)
			doc_lookup[len(all_passages)] = doc_id
			all_passages.append(passage)

		reranked_passages = rerank(query['question'], all_passages)
		rerank_hits = []
		seen_docs = set()
		for passage_score, passage_idx in reranked_passages:
			passage_doc = doc_lookup[passage_idx]
			if passage_doc not in seen_docs:
				rerank_hits.append((passage_doc, passage_score))
				seen_docs.add(passage_doc)

		for rank, (doc_id, score) in enumerate(rerank_hits, start=1):
			# smaller is better
			score = -score
			line = f'{query_id}\tQ0\t{doc_id}\t{rank}\t{score:.4f}\t{run_name}\n'
			fo.write(line)
print('Done!')
