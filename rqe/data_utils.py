
import os
import json
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import csv


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


def load_quora_data(data_path):
	examples = []
	with open(data_path, 'r') as f:
		csv_reader = csv.reader(f, delimiter='\t')
		for idx, row in enumerate(csv_reader):
			if idx == 0:
				continue
			# id	qid1	qid2	question1	question2	is_duplicate
			# A -> B
			example = {
				'id': row[0],
				'A': row[3].strip(),
				'B': row[4].strip(),
				'label': int(row[5])
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
		batch = {
			'input_ids': tokenizer_batch['input_ids'],
			'attention_mask': tokenizer_batch['attention_mask'],
			'token_type_ids': tokenizer_batch['token_type_ids'],
			'labels': labels,
			'ids': ids
		}

		return batch

