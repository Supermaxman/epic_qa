
import os
import json
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from tqdm import tqdm


class AnswerDataset(Dataset):
	def __init__(self, root_dir, input_path):
		self.root_dir = root_dir
		self.input_path = input_path
		self.ids = set()
		if self.input_path is not None:
			with open(self.input_path, 'r') as f:
				for line in f:
					line = line.strip()
					if line:
						q_id, _, full_id, rank, score, run_name = line.split()
						self.ids.add(full_id)
			self.doc_ids = defaultdict(lambda: defaultdict(set))
			for f_id in self.ids:
				start_id, end_id = f_id.split(':')
				doc_id, pass_id, sent_start_id = start_id.split('-')
				_, _, sent_end_id = end_id.split('-')
				self.doc_ids[doc_id][pass_id].add((sent_start_id, sent_end_id))
			self.examples = []
			for doc_id, doc_pass_ids in self.doc_ids.items():
				doc_path = os.path.join(self.root_dir, f'{doc_id}.json')
				with open(doc_path) as f:
					doc = json.load(f)
				for pass_id, sent_spans in doc_pass_ids.items():
					pass_idx = int(pass_id[1:])
					for sent_start_id, sent_end_id in sent_spans:
						sent_start_idx = int(sent_start_id[1:])
						sent_end_idx = int(sent_end_id[1:])
						passage = doc['contexts'][pass_idx]
						sentences = passage['sentences'][sent_start_idx:sent_end_idx+1]
						text = passage['text'][sentences[0]['start']:sentences[-1]['end']]
						example = {
							'id': f'{doc_id}-{pass_id}-{sent_start_id}:{doc_id}-{pass_id}-{sent_end_id}',
							'text': text
						}
						self.examples.append(example)
		else:
			self.ids = os.listdir(root_dir)
			self.examples = []
			for doc_name in tqdm(self.ids[:100]):
				doc_path = os.path.join(self.root_dir, doc_name)
				with open(doc_path) as f:
					doc = json.load(f)
				for passage in doc['contexts']:
					sentences = passage['sentences']
					start_sentence = sentences[0]
					end_sentence = sentences[-1]
					text = passage['text'][start_sentence['start']:end_sentence['end']]
					example = {
						'id': f'{start_sentence["sentence_id"]}:{end_sentence["sentence_id"]}',
						'text': text
					}
					self.examples.append(example)


	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		example = self.examples[idx]

		return example


class PredictionBatchCollator(object):
	def __init__(self, tokenizer,  max_seq_len: int, force_max_seq_len: bool):
		super().__init__()
		self.tokenizer = tokenizer
		self.max_seq_len = max_seq_len
		self.force_max_seq_len = force_max_seq_len

	def __call__(self, examples):
		# creates text examples
		ids = []
		sequences = []
		for ex in examples:
			ids.append(ex['id'])
			text = ex['text']
			sequences.append(f'{text} </s>')
		# "input_ids": batch["input_ids"].to(device),
		# "attention_mask": batch["attention_mask"].to(device),
		tokenizer_batch = self.tokenizer.batch_encode_plus(
			batch_text_or_text_pairs=sequences,
			padding='max_length' if self.force_max_seq_len else 'longest',
			return_tensors='pt',
			truncation=True,
			max_length=self.max_seq_len
		)
		batch = {
			'id': ids,
			'input_ids': tokenizer_batch['input_ids'],
			'attention_mask': tokenizer_batch['attention_mask'],
		}

		return batch
