
import os
import json
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import csv
import random

class ATPDataset(Dataset):
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


def load_smart_data(data_path):
	with open(data_path, 'r') as f:
		data = json.load(f)
	examples = []
	category_map = {}
	types_map = {}
	for ex in data:
		# A entails B (A -> B)
		# A is often more specific, B is more general
		ex_category = ex['category']
		if ex_category not in category_map:
			category_map[ex_category] = len(category_map)
		ex_category = category_map[ex_category]
		ex_types = []
		for ex_type in ex['type']:
			if ex_type != 'boolean':
				if ex_type not in types_map:
					types_map[ex_type] = len(types_map)
				ex_types.append(types_map[ex_type])
		example = {
			'id': ex['id'],
			'question': ex['question'].strip(),
			# single value like boolean, resource, literal, etc.
			'label': ex_category,
			# list of types
			'types': ex_types
		}
		examples.append(example)
	return examples, category_map, types_map


class BatchCollator(object):
	def __init__(self, tokenizer,  max_seq_len: int, force_max_seq_len: bool, types_map):
		super().__init__()
		self.tokenizer = tokenizer
		self.max_seq_len = max_seq_len
		self.force_max_seq_len = force_max_seq_len
		self.types_map = types_map

	def __call__(self, examples):
		# creates text examples
		sequences = []
		labels = []
		types = []
		ids = []
		for ex in examples:
			sequences.append(ex['question'])
			labels.append(ex['label'])
			ids.append(ex['id'])
			ex_type = torch.zeros(len(self.types_map), dtype=torch.float)
			for ex_t in ex['types']:
				ex_type[ex_t] = 1.0
			types.append(ex_type)
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
		types = torch.stack(types, dim=0)
		batch = {
			'input_ids': tokenizer_batch['input_ids'],
			'attention_mask': tokenizer_batch['attention_mask'],
			'token_type_ids': tokenizer_batch['token_type_ids'],
			'labels': labels,
			'types': types,
			'ids': ids
		}

		return batch

