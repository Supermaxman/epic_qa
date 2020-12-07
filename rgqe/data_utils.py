
import json
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import itertools


class PredictionBatchCollator(object):
	def __init__(self, tokenizer,  max_seq_len: int, force_max_seq_len: bool):
		super().__init__()
		self.tokenizer = tokenizer
		self.max_seq_len = max_seq_len
		self.force_max_seq_len = force_max_seq_len

	def __call__(self, examples):
		# creates text examples
		question_a_ids = []
		question_b_ids = []
		sample_ids = []
		sequences = []
		for ex in examples:
			question_a_ids.append(ex['question_a_id'])
			question_b_ids.append(ex['question_b_id'])
			sequences.append((ex['question_a_text'], ex['question_b_text']))
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
		batch = {
			'question_a_id': question_a_ids,
			'question_b_id': question_b_ids,
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


def make_id(sample):
	return f'{sample["question_id"]}|{sample["id"]}|{sample["sample_id"]}'


class RGQEPredictionDataset(Dataset):
	def __init__(self, input_path):
		self.input_path = input_path
		with open(input_path) as f:
			self.query_answers = json.load(f)

		self.examples = []
		for question_id, answers in self.query_answers.items():
			question_samples = []
			for answer in answers:
				answer_id = answer['answer_id']
				answer_samples = answer['samples']
				for sample in answer_samples:
					example = {
						'id': answer_id,
						'question_id': question_id,
						'sample_id':  sample['sample_id'],
						'sample_text': sample['sample_text']
					}
					question_samples.append(example)

			for sample_a, sample_b in itertools.combinations(question_samples, r=2):
				a_id = make_id(sample_a)
				b_id = make_id(sample_b)
				example = {
					'question_a_id': a_id,
					'question_b_id': b_id,
					'question_a_text': sample_a['sample_text'],
					'question_b_text': sample_b['sample_text'],
				}
				self.examples.append(example)

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		example = self.examples[idx]

		return example
