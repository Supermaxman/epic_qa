
import os
import json
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from rerank.sample_utils import NegativeSampler
import random


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


def find_ngrams(input_list, n):
	return zip(*[input_list[i:] for i in range(n)])


class QueryPassageDataset(Dataset):
	def __init__(self, root_dir, queries, multi_sentence, n_gram_max, document_qrels=None, passage_qrels=None):
		self.root_dir = root_dir
		self.file_names = os.listdir(root_dir)
		self.examples = []
		self.queries = queries
		self.multi_sentence = multi_sentence
		self.n_gram_max = n_gram_max
		self.document_qrels = document_qrels
		self.passage_qrels = passage_qrels

		for d_name in self.file_names:
			if not d_name.endswith('.json'):
				continue
			d_id = d_name.replace('.json', '')
			file_path = os.path.join(self.root_dir, d_name)
			with open(file_path) as f:
				doc = json.load(f)
				for query in queries:
					if document_qrels is not None and d_id not in document_qrels[query['question_id']]:
						continue
					for p_id, passage in enumerate(doc['contexts']):
						if passage_qrels is not None and f'{d_id}-{p_id}' not in passage_qrels[query['question_id']]:
							continue
						context_examples = []
						for s_id, sentence in enumerate(passage['sentences']):
							example = {
								'id': f'{d_id}-{p_id}-{s_id}-{s_id}',
								'question_id': query['question_id'],
								'd_id': d_id,
								'p_id': p_id,
								's_id': s_id,
								'text': passage['text'][sentence['start']:sentence['end']],
								'query': query['question']
							}
							self.examples.append(example)
							context_examples.append(example)
						if self.multi_sentence:
							# generate sentence n-grams from 2 to n_gram_max of contiguous sentence spans
							for k in range(2, self.n_gram_max+1):
								for ex_list in find_ngrams(context_examples, n=k):
									ex_first = ex_list[0]
									ex_last = ex_list[-1]
									example = {
										'id': f'{ex_first["d_id"]}-{ex_first["p_id"]}-{ex_first["s_id"]}-{ex_last["s_id"]}',
										'question_id': ex_first['question_id'],
										'd_id': ex_first['d_id'],
										'p_id': ex_first['p_id'],
										's_id': f'{ex_first["s_id"]}-{ex_last["s_id"]}',
										'text': ' '.join([ex['text'] for ex in ex_list]),
										'query': ex_first['query']
									}
									self.examples.append(example)

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		example = self.examples[idx]

		return example


def split_data(data, train_ratio=0.8):
	random.shuffle(data)
	train_size = int(len(data) * train_ratio)
	train_data = data[:train_size]
	dev_data = data[train_size:]
	return train_data, dev_data


def load_expert_data(data_path):
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


def load_consumer_data(data_path):
	examples = []
	queries = []
	answers = []
	# root folders which contain categories
	for folder_name in os.listdir(data_path):
		folder_path = os.path.join(data_path, folder_name)
		if os.path.isdir(folder_path):
			# xml files with query answer pairs
			for file_name in os.listdir(folder_path):
				if file_name.endswith('.xml'):
					file_path = os.path.join(folder_path, file_name)
					tree = ET.parse(file_path)
					qa_pairs = tree.getroot()[2]
					for q_xml, a_xml in qa_pairs:
						q_text = q_xml.text
						a_text = a_xml.text
						query = {
							'id': hash(q_text),
							'text': q_text
						}
						queries.append(query)
						answer = {
							'id': hash(a_text),
							'text': a_text
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
		sequences = []
		for ex in examples:
			ids.append(ex['id'])
			question_ids.append(ex['question_id'])
			sequences.append((ex['query'], ex['text']))
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
			'input_ids': tokenizer_batch['input_ids'],
			'attention_mask': tokenizer_batch['attention_mask'],
			'token_type_ids': tokenizer_batch['token_type_ids'],
		}

		return batch
