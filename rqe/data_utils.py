
import os
import json
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import csv
import random


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
	def __init__(self, tokenizer,  max_seq_len: int, force_max_seq_len: bool, category_map, types_map):
		super().__init__()
		self.tokenizer = tokenizer
		self.max_seq_len = max_seq_len
		self.force_max_seq_len = force_max_seq_len
		self.category_map = category_map
		self.types_map = types_map

	def __call__(self, examples):
		# creates text examples
		sequences = []
		labels = []
		ids = []
		a_categories = []
		b_categories = []
		a_types = []
		b_types = []
		for ex in examples:
			sequences.append((ex['A'], ex['B']))
			labels.append(ex['label'])
			ids.append(ex['id'])
			a_categories.append(self.category_map[ex['A_category']])
			b_categories.append(self.category_map[ex['B_category']])
			a_type = torch.zeros(len(self.types_map), dtype=torch.float)
			b_type = torch.zeros(len(self.types_map), dtype=torch.float)
			for ex_t in ex['A_types']:
				a_type[self.types_map[ex_t]] = 1.0
			a_types.append(a_type)
			for ex_t in ex['B_types']:
				b_type[self.types_map[ex_t]] = 1.0
			b_types.append(b_type)

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
		a_categories = torch.tensor(a_categories, dtype=torch.long)
		b_categories = torch.tensor(b_categories, dtype=torch.long)
		a_types = torch.stack(a_types, dim=0)
		b_types = torch.stack(b_types, dim=0)
		batch = {
			'input_ids': tokenizer_batch['input_ids'],
			'attention_mask': tokenizer_batch['attention_mask'],
			'token_type_ids': tokenizer_batch['token_type_ids'],
			'labels': labels,
			'ids': ids,
			'A_categories': a_categories,
			'B_categories': b_categories,
			'A_types': a_types,
			'B_types': b_types,
		}

		return batch

