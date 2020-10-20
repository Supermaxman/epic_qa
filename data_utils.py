
import os
import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from sample_utils import NegativeSampler


class QuestionAnswerDataset(Dataset):
	def __init__(self, examples):
		self.examples = examples

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		ex = self.examples[idx]

		return ex


def split_data(data, train_ratio=0.8):
	train_size = int(len(data) * train_ratio)
	train_data = data[:train_size]
	dev_data = data[train_size:]
	return train_data, dev_data


def load_data(data_path):
	examples = []
	queries = []
	answers = []
	for file_name in os.listdir(data_path):
		if file_name.endswith('.json'):
			file_path = os.path.join(data_path, file_name)
			with open(file_path, 'r') as f:
				data = json.load(f)
			for question in tqdm(data['questions'], desc=f'{file_name}'):
				query = {
					'id': question['id'],
					'text': question['body']
				}
				queries.append(query)
				for snippet in question['snippets']:
					answer = {
						'id': f"{snippet['document']}-{snippet['beginSection']}-{snippet['endSection']}-"
						f"{snippet['offsetInBeginSection']}-{snippet['offsetInEndSection']}",
						'text': snippet['text']
					}
					answers.append(answer)
					example = {
						'query': query,
						'answer': answer
					}
					examples.append(example)
	return examples, queries, answers


class SampleCollator(object):
	def __init__(self, tokenizer, neg_sampler: NegativeSampler, max_seq_len: int, force_max_seq_len: bool):
		super().__init__()
		self.tokenizer = tokenizer
		self.neg_sampler = neg_sampler
		self.max_seq_len = max_seq_len
		self.force_max_seq_len = force_max_seq_len

	def __call__(self, pos_examples):
		# creates text examples
		batch_negative_sample_size = None
		sequences = []
		labels = []
		for pos_ex in pos_examples:
			sequences.append((pos_ex['query']['text'], pos_ex['answer']['text']))
			labels.append(1)
			num_samples = 0
			for neg_ex in self.neg_sampler.sample(pos_ex):
				sequences.append((neg_ex['query']['text'], neg_ex['answer']['text']))
				labels.append(0)
				num_samples += 1
			if batch_negative_sample_size is None:
				batch_negative_sample_size = num_samples

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
		labels = torch.tensor(labels, dtype=torch.long)
		sample_size = batch_negative_sample_size + 1
		batch = {
			'input_ids': tokenizer_batch['input_ids'],
			'attention_mask': tokenizer_batch['attention_mask'],
			'token_type_ids': tokenizer_batch['token_type_ids'],
			'labels': labels,
			'sample_size': sample_size
		}

		return batch
