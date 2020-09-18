import json
from tqdm import tqdm
import argparse
from collections import defaultdict
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader


def find_ngrams(input_list, n):
	return zip(*[input_list[i:] for i in range(n)])


class Sentence(object):
	def __init__(self, id, text):
		self.id = id
		self.text = text


class SentenceSpan(object):
	def __init__(self, sentences, parent=None):
		self.sentences = sentences
		self.parent = parent
		self.children = []
		self.score = float('-inf')
		self.sibling = None

	@property
	def text(self):
		return ' '.join([s.text for s in self.sentences])

	@property
	def id(self):
		first_sentence = self.sentences[0]
		last_sentence = self.sentences[-1]
		return f'{first_sentence.id}:{last_sentence.id}'

	def split(self, idx):
		assert idx > 0, 'Split must not leave left list empty'
		assert idx < len(self.sentences), 'Split must not leave right list empty'
		left = SentenceSpan(sentences=self.sentences[:idx], parent=self)
		right = SentenceSpan(sentences=self.sentences[idx:], parent=self)
		left.sibling = right
		right.sibling = left
		self.children.append(left)
		self.children.append(right)
		return left, right

	def full_split(self):
		splits = []
		for i in range(1, len(self)):
			left, right = self.split(i)
			splits.append(left)
			splits.append(right)
		return splits

	def __len__(self):
		return len(self.sentences)

	def __eq__(self, other):
		if not isinstance(other, SentenceSpan):
			return False
		if other.id == self.id:
			return True
		else:
			return False

	def __hash__(self):
		return hash(self.id)

	def __str__(self):
		return f'{self.id}\t{self.text}'

	def __repr__(self):
		return str(self)


def load_passages(root_dir, file_names):
	passages = []
	for d_id in file_names:
		file_path = os.path.join(root_dir, d_id + '.json')
		with open(file_path) as f:
			doc = json.load(f)
			for p_id, passage in enumerate(doc['contexts']):
				sentence_list = []
				for s_id, sentence in enumerate(passage['sentences']):
					s = Sentence(
						id=sentence['sentence_id'],
						text=passage['text'][sentence['start']:sentence['end']]
					)
					sentence_list.append(s)
				p = SentenceSpan(sentence_list)
				passages.append(p)
	return passages


class SentenceSpanDataset(Dataset):
	def __init__(self, examples, query):
		for ex in examples:
			ex.query = query
		self.examples = examples
		self.query = query

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
parser.add_argument('-bs', '--batch_size', default=64, type=int)
parser.add_argument('-ml', '--max_length', default=512, type=int)
parser.add_argument('-k', '--top_k', default=1000, type=int)

args = parser.parse_args()

# 'expert'
collection_path = args.collection_path
label_path = args.label_path
# 'baseline_doc'
run_name = args.run_path
# expert_questions_prelim.json
query_path = args.query_path
run_path = args.run_path
top_k = args.top_k

rerank_model_name = args.rerank_model
batch_size = args.batch_size
max_length = args.max_length

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

print(f'Loading tokenizer: {rerank_model_name}')
tokenizer = AutoTokenizer.from_pretrained(rerank_model_name)

print(f'Loading model: {rerank_model_name}')
model = AutoModelForSequenceClassification.from_pretrained(rerank_model_name)
model.to(device)
model.eval()


def collate_batch(examples):
	passages = [x.text for x in examples]
	queries = [x.query for x in examples]
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
	batch['id'] = [x.id for x in examples]
	batch['examples'] = examples
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
pass_scores = {}
for query_idx, query in tqdm(enumerate(queries, start=1), total=len(queries)):
	query_labels = qrels[query_idx]
	query_id = query['question_id']
	assert len(query_labels) > 0

	example_scores = {}
	# first we compute scores for all depth-0 (passages) sentence spans
	all_examples = []
	current_examples = load_passages(collection_path, query_labels)
	seen_examples = set()
	prev_examples = None
	depth = 0
	while True:
		dataset = SentenceSpanDataset(
			current_examples,
			query['question']
		)
		dataloader = DataLoader(
			dataset,
			batch_size=batch_size,
			shuffle=False,
			num_workers=1,
			collate_fn=collate_batch
		)

		for batch in tqdm(dataloader, total=len(dataloader), desc=f'{query_id} Depth {depth}'):
			with torch.no_grad():
				scores = model(
					input_ids=batch['input_ids'].to(device),
					token_type_ids=batch['token_type_ids'].to(device),
					attention_mask=batch['attention_mask'].to(device)
				)[0][:, 0].cpu().numpy()
			for span, score in zip(batch['examples'], scores):
				# make larger score mean better answer
				example_scores[span] = -score

		# remove parent examples ranking lower than children
		removed_examples = set()
		for example in current_examples:
			# example is a child so we must consider pruning parent
			if example.parent is not None:
				if example_scores[example.parent] < example_scores[example]:
					# remove parent, child is better answer
					removed_examples.add(example.parent)
				elif example_scores[example.parent] < example_scores[example.sibling]:
					# parent has better answer than child, but sibling better than parent.
					# We prefer shorter answers (sibling) and it is better, so remove parent in favor of better
					# sibling and worse child
					removed_examples.add(example.parent)
				else:
					# neither sibling nor child have better score than parent, so prune.
					removed_examples.add(example)

		all_examples += current_examples

		# sort examples by highest to lowest score
		# TODO make more efficient
		# TODO make more efficient
		all_examples = [ex for ex in all_examples if ex not in removed_examples]
		all_examples = list(sorted(all_examples, key=lambda x: example_scores[x], reverse=True))

		# next we eliminate spans with less than top-k score:
		all_examples = all_examples[:top_k]
		lowest_score = example_scores[all_examples[-1]]
		# finally we split spans within top-k
		next_examples = []
		for span in current_examples:
			if example_scores[span] >= lowest_score:
				span_splits = span.full_split()
				for child in span_splits:
					if child not in example_scores:
						next_examples.append(child)

		if len(next_examples) == 0:
			break
		current_examples = next_examples
		depth += 1
	print(f'{query_id}: {len(all_examples)}')
	for example in all_examples:
		example.score = example_scores[example]

	pass_scores[query_id] = all_examples


# TODO sort
print(f'Saving results...')
with open(run_path, 'w') as fo:
	for query_id, spans in pass_scores.items():
		for rank, span in enumerate(spans, start=1):
			line = f'{query_id}\tQ0\t{span.id}\t{rank}\t{span.score:.4f}\t{run_name}\n'
			fo.write(line)
print('Done!')
