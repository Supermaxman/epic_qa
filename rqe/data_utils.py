
import os
import json
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import csv
import random
from collections import defaultdict


class RQEDataset(Dataset):
	def __init__(self, examples):
		self.examples = examples

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		ex = self.examples[idx]

		return ex


def split_data(data, ratio=0.8):
	random.shuffle(data)
	train_size = int(len(data) * ratio)
	train_data = data[:train_size]
	dev_data = data[train_size:]
	return train_data, dev_data


def load_clinical_data(data_path):
	tree = ET.parse(data_path)
	pairs = tree.getroot()

	examples = []
	for pair in pairs:
		# A entails B (A -> B)
		# A is often more specific, B is more general
		example = {
			'id': pair.get('pid'),
			'A': pair[0].text.strip(),
			'B': pair[1].text.strip(),
			'label': 1 if pair.get('value').lower() == 'true' else 0
		}
		examples.append(example)
	return examples


def load_smart_maps(category_map_path, types_map_path):
	with open(category_map_path, 'r') as f:
		category_map = json.load(f)
	with open(types_map_path, 'r') as f:
		types_map = json.load(f)
	return category_map, types_map


def load_at_predictions(predictions_path):
	pred_list = torch.load(predictions_path)
	predictions = {pred['id']: pred for pred in pred_list}
	return predictions


def load_q_hier_data(data_path, neg_samples=None):
	with open(data_path) as f:
		data = json.load(f)

	pos_examples = data['pos_examples']
	neg_examples = data['neg_examples']
	if neg_samples is not None:
		random.shuffle(neg_examples)
		neg_examples = neg_examples[:neg_samples]
	examples = []
	for p_example in pos_examples + neg_examples:
		example = {
			'id': p_example['id'],
			'A': p_example['super_question'],
			'B': p_example['sub_question'],
			'label': p_example['label']
		}
		examples.append(example)

	return examples


def load_quora_data(data_path, at_predictions):
	examples = []
	with open(data_path, 'r') as f:
		csv_reader = csv.reader(f, delimiter='\t')
		for idx, row in enumerate(csv_reader):
			if idx == 0:
				continue
			# id	qid1	qid2	question1	question2	is_duplicate
			# A -> B
			ex_id = row[0]
			a_txt = row[3].strip()
			a_id = f'{ex_id}|A'
			a_predictions = at_predictions[a_id]
			a_category, a_types = a_predictions['category'], a_predictions['types']
			b_txt = row[4].strip()
			b_id = f'{ex_id}|B'
			b_predictions = at_predictions[b_id]
			b_category, b_types = b_predictions['category'], b_predictions['types']
			label = int(row[5])
			example = {
				'id': ex_id,
				'A': a_txt,
				'A_category': a_category,
				'A_types': a_types,
				'B': b_txt,
				'B_category': b_category,
				'B_types': b_types,
				'label': label
			}
			examples.append(example)
	return examples


class BatchCollator(object):
	def __init__(self, tokenizer,  max_seq_len: int, force_max_seq_len: bool):
		super().__init__()
		self.tokenizer = tokenizer
		self.max_seq_len = max_seq_len
		self.force_max_seq_len = force_max_seq_len

	def __call__(self, examples):
		# creates text examples
		sequences = []
		labels = []
		ids = []
		for ex in examples:
			sequences.append((ex['A'], ex['B']))
			labels.append(ex['label'])
			ids.append(ex['id'])

		# "input_ids": batch["input_ids"].to(device),
		# "attention_mask": batch["attention_mask"].to(device),
		tokenizer_batch = self.tokenizer.batch_encode_plus(
			batch_text_or_text_pairs=sequences,
			add_special_tokens=True,
			padding='max_length' if self.force_max_seq_len else 'longest',
			return_tensors='pt',
			truncation=True,
			max_length=self.max_seq_len
		)
		labels = torch.tensor(labels, dtype=torch.long)
		input_ids = tokenizer_batch['input_ids']
		attention_mask = tokenizer_batch['attention_mask']
		type_ids = tokenizer_batch['token_type_ids']

		a_mask = (type_ids.eq(0).bool() & attention_mask.bool()).long()
		b_mask = (type_ids.eq(1).bool() & attention_mask.bool()).long()

		batch = {
			'input_ids': input_ids,
			'attention_mask': attention_mask,
			'token_type_ids': type_ids,
			'labels': labels,
			'ids': ids,
			'A_mask': a_mask,
			'B_mask': b_mask,
		}

		return batch


class PredictionBatchCollator(object):
	def __init__(self, tokenizer,  max_seq_len: int, force_max_seq_len: bool):
		super().__init__()
		self.tokenizer = tokenizer
		self.max_seq_len = max_seq_len
		self.force_max_seq_len = force_max_seq_len

	def __call__(self, examples):
		# creates text examples
		ids = []
		question_ids = []
		sample_ids = []
		sequences = []
		for ex in examples:
			ids.append(ex['id'])
			question_ids.append(ex['question_id'])
			sample_ids.append(ex['sample_id'])
			sequences.append((ex['sample'], ex['query']))
		# "input_ids": batch["input_ids"].to(device),
		# "attention_mask": batch["attention_mask"].to(device),
		tokenizer_batch = self.tokenizer.batch_encode_plus(
			batch_text_or_text_pairs=sequences,
			add_special_tokens=True,
			padding='max_length' if self.force_max_seq_len else 'longest',
			return_tensors='pt',
			truncation='only_second',
			max_length=self.max_seq_len
		)
		batch = {
			'id': ids,
			'question_id': question_ids,
			'sample_id': sample_ids,
			'input_ids': tokenizer_batch['input_ids'],
			'attention_mask': tokenizer_batch['attention_mask'],
			'token_type_ids': tokenizer_batch['token_type_ids'],
		}

		return batch


def load_run(run_path):
	query_answers = defaultdict(list)
	with open(run_path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				q_id, _, answer_id, rank, score, run_name = line.split()
				query_answers[q_id].append(answer_id)
	return query_answers


class RQEPredictionDataset(Dataset):
	def __init__(self, expand_path, input_path, queries):
		self.expand_path = expand_path
		self.input_path = input_path
		self.query_lookup = {query['question_id']: query for query in queries}

		with open(self.expand_path, 'r') as f:
			self.answer_samples = json.load(f)

		self.query_answers = load_run(self.input_path)

		self.examples = []
		for question_id, answer_ids in self.query_answers.items():
			query = self.query_lookup[question_id]
			query_text = query['question']
			for answer_id in answer_ids:
				answer_samples = self.answer_samples[answer_id]
				for sample_id, sample in enumerate(answer_samples):
					example = {
						'id': answer_id,
						'question_id': question_id,
						'sample_id':  sample_id,
						'sample': sample,
						'query': query_text
					}
					self.examples.append(example)

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		example = self.examples[idx]

		return example