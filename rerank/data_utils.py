
import os
import json
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from sample_utils import NegativeSampler
import random
from collections import defaultdict


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
	def __init__(self, root_dir, queries, multi_sentence, n_gram_max, document_qrels=None, passage_qrels=None, only_passages=False):
		self.root_dir = root_dir
		self.query_lookup = {query['question_id']: query for query in queries}
		assert not (document_qrels is not None and passage_qrels is not None), 'Cannot specify both doc and pass qrels!'
		if document_qrels is not None:
			file_names = set()
			self.query_docs = defaultdict(set)
			self.query_doc_pass = None
			for question_id, question_files in document_qrels.items():
				if question_id not in self.query_lookup:
					continue
				for doc_id in question_files:
					file_names.add(f'{doc_id}.json')
					self.query_docs[doc_id].add(question_id)
			self.file_names = sorted(list(file_names))

		elif passage_qrels is not None:
			file_names = set()
			self.query_docs = defaultdict(set)
			self.query_doc_pass = defaultdict(lambda: defaultdict(set))
			for question_id, question_files in passage_qrels.items():
				if question_id not in self.query_lookup:
					continue
				for doc_pass_id in question_files:
					doc_id = '-'.join(doc_pass_id.split('-')[:-1])
					file_names.add(f'{doc_id}.json')
					self.query_docs[doc_id].add(question_id)
					self.query_doc_pass[question_id][doc_id].add(doc_pass_id)
			self.file_names = sorted(list(file_names))

		else:
			self.file_names = os.listdir(root_dir)
			self.query_docs = None
			self.query_doc_pass = None
		self.examples = []
		self.queries = queries
		self.multi_sentence = multi_sentence
		self.n_gram_max = n_gram_max
		self.only_passages = only_passages
		warned = False
		self.question_example_count = defaultdict(int)
		for d_name in tqdm(self.file_names):
			if not d_name.endswith('.json'):
				continue
			d_id = d_name.replace('.json', '')
			file_path = os.path.join(self.root_dir, d_name)
			if not os.path.exists(file_path):
				if not warned:
					print('WARNING: some missing files')
					warned = True
				continue
			with open(file_path, 'r') as f:
				doc = json.load(f)
			if self.query_docs is None:
				doc_queries = queries
			else:
				doc_queries = [self.query_lookup[q_id] for q_id in sorted(self.query_docs[d_id])]
			for query in doc_queries:
				question_id = query['question_id']
				if self.query_doc_pass is None:
					doc_contexts = doc['contexts']
				else:
					context_lookup = {c['context_id']: c for c in doc['contexts']}
					doc_contexts = []
					for p_id in sorted(self.query_doc_pass[question_id][d_id]):
						if p_id in context_lookup:
							doc_contexts.append(context_lookup[p_id])
						else:
							if not warned:
								print(f'{p_id} not found in context: {context_lookup}')
								print('WARNING: some missing contexts')
								warned = True

				for passage in doc_contexts:
					context_examples = []
					if not self.only_passages:
						for sentence in passage['sentences']:
							s_id = sentence['sentence_id']
							example = {
								'id': f'{s_id}:{s_id}',
								'question_id': question_id,
								's_id': s_id,
								'text': passage['text'][sentence['start']:sentence['end']],
								'query': query['question']
								# 'query': query['question'] + ' ' + query['query'] + ', ' + query['background']
							}
							self.examples.append(example)
							self.question_example_count[question_id] += 1
							context_examples.append(example)
						if self.multi_sentence:
							# generate sentence n-grams from 2 to n_gram_max of contiguous sentence spans
							for k in range(2, self.n_gram_max+1):
								for ex_list in find_ngrams(context_examples, n=k):
									ex_first = ex_list[0]
									ex_last = ex_list[-1]
									example = {
										'id': f'{ex_first["s_id"]}:{ex_last["s_id"]}',
										'question_id': ex_first['question_id'],
										's_id': f'{ex_first["s_id"]}:{ex_last["s_id"]}',
										'text': ' '.join([ex['text'] for ex in ex_list]),
										'query': ex_first['query']
									}
									self.examples.append(example)
					else:
						start_sentence = passage['sentences'][0]
						end_sentence = passage['sentences'][-1]
						start_s_id = start_sentence["sentence_id"]
						end_s_id = end_sentence["sentence_id"]
						example = {
							'id': f'{start_s_id}:{end_s_id}',
							'question_id': question_id,
							's_id': f'{start_s_id}:{end_s_id}',
							'text': passage['text'][start_sentence['start']:end_sentence['end']],
							'query': query['question']
							# 'query': query['question'] + ' ' + query['query'] + ', ' + query['background']
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


class RerankBatchCollator(object):
	def __init__(self, tokenizer,  max_seq_len: int, force_max_seq_len: bool):
		super().__init__()
		self.tokenizer = tokenizer
		self.max_seq_len = max_seq_len
		self.force_max_seq_len = force_max_seq_len

	def __call__(self, examples):
		ids = []
		labels = []
		weights = []
		question_ids = []
		sequences = []
		for ex in examples:
			ids.append(ex['id'])
			labels.append(ex['label'])
			weights.append(ex['weight'])
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
			'labels': torch.tensor(labels, dtype=torch.long),
			'weights': torch.tensor(weights, dtype=torch.float),
			'input_ids': tokenizer_batch['input_ids'],
			'attention_mask': tokenizer_batch['attention_mask'],
			'token_type_ids': tokenizer_batch['token_type_ids'],
		}

		return batch


class QueryPassageLabeledDataset(Dataset):
	def __init__(self, root_dir, queries, labels, multi_sentence, n_gram_max, document_qrels=None, passage_qrels=None, only_passages=False,
		negative_samples=10000, add_all_labels=False):
		self.root_dir = root_dir
		self.negative_samples = negative_samples
		self.labels = {}
		self.add_all_labels = add_all_labels
		for label in labels:
			question_id = label['question_id']
			annotations = {a['sentence_id']: a for a in label['annotations']}
			label['annotations'] = annotations
			self.labels[question_id] = label

		self.query_lookup = {query['question_id']: query for query in queries}
		assert not (document_qrels is not None and passage_qrels is not None), 'Cannot specify both doc and pass qrels!'
		if document_qrels is not None:
			file_names = set()
			self.query_docs = defaultdict(set)
			self.query_doc_pass = None
			for question_id, question_files in document_qrels.items():
				if question_id not in self.query_lookup:
					continue
				for doc_id in question_files:
					file_names.add(f'{doc_id}.json')
					self.query_docs[doc_id].add(question_id)
			self.file_names = list(file_names)

		elif passage_qrels is not None:
			file_names = set()
			self.query_docs = defaultdict(set)
			self.query_doc_pass = defaultdict(lambda: defaultdict(set))
			for question_id, question_files in passage_qrels.items():
				if question_id not in self.query_lookup:
					continue
				if question_id not in self.labels:
					continue
				if self.add_all_labels:
					for sentence_id in self.labels[question_id]['annotations']:
						doc_id, pass_id, sent_id = sentence_id.split('-')
						file_names.add(f'{doc_id}.json')
						self.query_docs[doc_id].add(question_id)
						self.query_doc_pass[question_id][doc_id].add(f'{doc_id}-{pass_id}')

				for doc_pass_id in question_files:
					doc_id, pass_id = doc_pass_id.split('-')
					file_names.add(f'{doc_id}.json')
					self.query_docs[doc_id].add(question_id)
					self.query_doc_pass[question_id][doc_id].add(doc_pass_id)
			self.file_names = sorted(list(file_names))

		else:
			self.file_names = sorted(os.listdir(root_dir))
			self.query_docs = None
			self.query_doc_pass = None
		self.examples = []
		self.queries = queries
		self.multi_sentence = multi_sentence
		self.n_gram_max = n_gram_max
		self.only_passages = only_passages
		self.num_positive = 0
		self.num_negative = 0
		self.question_negative_examples = defaultdict(list)
		warned = False
		for d_name in tqdm(self.file_names):
			if not d_name.endswith('.json'):
				continue
			d_id = d_name.replace('.json', '')
			file_path = os.path.join(self.root_dir, d_name)
			if not os.path.exists(file_path):
				if not warned:
					print('WARNING: some missing files')
					warned = True
				continue
			with open(file_path, 'r') as f:
				doc = json.load(f)
			if self.query_docs is None:
				doc_queries = queries
			else:
				doc_queries = [self.query_lookup[q_id] for q_id in self.query_docs[d_id]]
			for query in doc_queries:
				question_id = query['question_id']
				if question_id not in self.labels:
					continue
				q_labels = self.labels[question_id]['annotations']
				if self.query_doc_pass is None:
					doc_contexts = doc['contexts']
				else:
					context_lookup = {c['context_id']: c for c in doc['contexts']}
					doc_contexts = []
					for p_id in self.query_doc_pass[question_id][d_id]:
						if p_id in context_lookup:
							doc_contexts.append(context_lookup[p_id])
						else:
							if not warned:
								print(f'{p_id} not found in context: {context_lookup}')
								print('WARNING: some missing contexts')
								warned = True

				for passage in doc_contexts:
					context_examples = []
					if not self.only_passages:
						for sentence in passage['sentences']:
							s_id = sentence['sentence_id']
							if s_id in q_labels:
								weight = float(len(q_labels[s_id]['nugget_ids']))
								label = 1
								self.num_positive += 1
							else:
								weight = 1.0
								label = 0

							example = {
								'id': f'{s_id}:{s_id}',
								'question_id': question_id,
								's_id': s_id,
								'text': passage['text'][sentence['start']:sentence['end']],
								'query': query['question'],
								'weight': weight,
								'label': label
							}
							if label == 1:
								self.examples.append(example)
							else:
								self.question_negative_examples[question_id].append(example)
							context_examples.append(example)
						if self.multi_sentence:
							raise NotImplementedError()
							# generate sentence n-grams from 2 to n_gram_max of contiguous sentence spans
							for k in range(2, self.n_gram_max+1):
								for ex_list in find_ngrams(context_examples, n=k):
									ex_first = ex_list[0]
									ex_last = ex_list[-1]
									example = {
										'id': f'{ex_first["s_id"]}:{ex_last["s_id"]}',
										'question_id': ex_first['question_id'],
										's_id': f'{ex_first["s_id"]}:{ex_last["s_id"]}',
										'text': ' '.join([ex['text'] for ex in ex_list]),
										'query': ex_first['query']
									}
									self.examples.append(example)
					else:
						raise NotImplementedError()
						start_sentence = passage['sentences'][0]
						end_sentence = passage['sentences'][-1]
						start_s_id = start_sentence["sentence_id"]
						end_s_id = end_sentence["sentence_id"]
						example = {
							'id': f'{start_s_id}:{end_s_id}',
							'question_id': question_id,
							's_id': f'{start_s_id}:{end_s_id}',
							'text': passage['text'][start_sentence['start']:end_sentence['end']],
							'query': query['question']
							# 'query': query['question'] + ' ' + query['query'] + ', ' + query['background']
						}
		self.negative_examples = []
		for question_id, q_negative_examples in self.question_negative_examples.items():
			random.shuffle(q_negative_examples)
			q_negative_examples = q_negative_examples[:self.negative_samples]
			self.negative_examples += q_negative_examples

		self.num_negative = len(self.negative_examples)
		self.examples = self.examples + self.negative_examples
		random.shuffle(self.examples)

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		example = self.examples[idx]

		return example

