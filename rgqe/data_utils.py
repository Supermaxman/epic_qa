
import json
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import itertools
import os


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


class RGQESelfPredictionDataset(Dataset):
	def __init__(self, input_path):
		self.input_path = input_path
		with open(input_path) as f:
			# [answer_id] -> samples
			self.answers = json.load(f)

		self.examples = []
		for answer_id, a_samples in self.answers.items():
			question_samples = []
			for sample_id, a_sample_text in enumerate(a_samples):
				example = {
					'id': f'{answer_id}|{sample_id}',
					'sample_text': a_sample_text
				}
				question_samples.append(example)

			for sample_a, sample_b in itertools.combinations(question_samples, r=2):
				example = {
					'question_a_id': sample_a['id'],
					'question_b_id': sample_b['id'],
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


class RGQEQuestionPredictionDataset(Dataset):
	def __init__(self, input_path, search_path, queries, top_k):
		self.input_path = input_path
		self.search_path = search_path
		self.top_k = top_k
		self.queries = {q['question_id']: q for q in queries}

		with open(input_path) as f:
			self.answers = json.load(f)

		question_answer_count = defaultdict(int)
		answer_questions = defaultdict(list)
		with open(self.search_path, 'r') as f:
			for line in f:
				line = line.strip()
				if line:
					question_id, _, answer_id, rank, score, run_name = line.split()
					if question_answer_count[question_id] > top_k:
						continue
					question_answer_count[question_id] += 1
					answer_questions[answer_id].append(question_id)

		self.examples = []
		for answer_id, a_sets in self.answers.items():
			for question_id in answer_questions[answer_id]:
				question_text = self.queries[question_id]['question']
				for entailed_set in a_sets:
					entailed_set_id = entailed_set['entailed_set_id']
					entailed_set_sample_text = entailed_set['entailed_set'][0]['sample_text']
					example = {
						'question_a_id': f'{answer_id}|{entailed_set_id}',
						'question_b_id': question_id,
						'question_a_text': entailed_set_sample_text,
						'question_b_text': question_text,
					}
					self.examples.append(example)

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		example = self.examples[idx]

		return example


class RGQETopPredictionDataset(Dataset):
	def __init__(self, input_path, qe_path, rr_path, search_path, top_k, qe_threshold, rr_threshold):
		self.input_path = input_path
		self.search_path = search_path
		self.qe_path = qe_path
		self.rr_path = rr_path
		self.top_k = top_k

		with open(input_path) as f:
			# [answer_id] -> entailed sets
			self.answers = json.load(f)
		if qe_path is not None:
			with open(qe_path) as f:
				# [question_id][answer_id] -> (entailed set, entailed_prob)
				self.qa_set_entailments = json.load(f)
		if rr_path is not None:
			with open(rr_path) as f:
				# [answer_id][str(entailed_set_id)] -> rank
				self.rr_ranks = json.load(f)

		question_answer_count = defaultdict(int)
		question_samples = defaultdict(list)
		with open(self.search_path) as f:
			for line in f:
				line = line.strip()
				if line:
					question_id, _, answer_id, rank, score, run_name = line.split()
					if question_answer_count[question_id] >= top_k:
						continue
					question_answer_count[question_id] += 1
					answer_sets = self.answers[answer_id]
					for entailed_set in answer_sets:
						entailed_set_id = entailed_set['entailed_set_id']
						if qe_path is not None:
							entail_prob = self.qa_set_entailments[question_id][answer_id][entailed_set_id]
							if entail_prob < qe_threshold:
								continue
						if rr_path is not None:
							es_rank = self.rr_ranks[answer_id][str(entailed_set_id)]
							if es_rank < rr_threshold:
								continue
						entailed_set_sample_text = entailed_set['entailed_set'][0]['sample_text']
						example = {
							'id': f'{question_id}|{answer_id}|{entailed_set_id}',
							'answer_id': answer_id,
							'entailed_set_text': entailed_set_sample_text
						}
						question_samples[question_id].append(example)

		self.examples = []
		for question_id, q_samples in question_samples.items():
			print(f'{question_id}: {len(q_samples)}')
			for sample_a, sample_b in itertools.combinations(q_samples, r=2):
				# ignore self entailment since that was already computed
				if sample_a['answer_id'] == sample_b['answer_id']:
					continue
				example = {
					'question_a_id': sample_a['id'],
					'question_b_id': sample_b['id'],
					'question_a_text': sample_a['entailed_set_text'],
					'question_b_text': sample_b['entailed_set_text'],
				}
				self.examples.append(example)

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		example = self.examples[idx]

		return example


class RGQEAllPredictionDataset(Dataset):
	def __init__(self, input_path, qe_path, cc_path, threshold):
		self.input_path = input_path
		self.qe_path = qe_path
		self.cc_path = cc_path
		self.threshold = threshold

		with open(self.input_path, 'r') as f:
			self.top_cc = json.load(f)

		with open(self.cc_path, 'r') as f:
			# [answer_id] -> list of entailed sets
			self.answer_sets = json.load(f)

		self.q_entailed_sets = defaultdict(lambda: defaultdict(set))
		with open(self.qe_path, 'r') as f:
			# [answer_id] -> list of entailed sets
			qa_set_entailments = json.load(f)
			for question_id, q_set_entailments in qa_set_entailments.items():
				for answer_id, a_set_entailments in q_set_entailments.items():
					for entailed_set_id, entail_prob in a_set_entailments:
						if entail_prob < threshold:
							continue
						self.q_entailed_sets[question_id][answer_id].add(entailed_set_id)
		self.examples = []
		for question_id, q_top_cc in self.top_cc.items():
			# list of entailed_set_id, entailed_set with entailed_set containing samples,
			# where entailed_set[0]['entailed_set_text'] is representative question
			entailed_sets = q_top_cc['entailed_sets']
			# [answer_id] -> merged_entailed_sets for top_k answers
			top_answer_sets = q_top_cc['answer_sets']
			num_sets = 0
			num_possible_sets = 0
			for answer_id, a_sets in self.answer_sets.items():
				if answer_id in top_answer_sets:
					continue
				for entailed_set_a in a_sets:
					entailed_set_a_id = entailed_set_a['entailed_set_id']
					num_possible_sets += 1
					if entailed_set_a_id not in self.q_entailed_sets[question_id][answer_id]:
						continue
					entailed_set_a_text = entailed_set_a['entailed_set'][0]['sample_text']
					num_sets += 1
					for entailed_set_b in entailed_sets:
						entailed_set_b_id = entailed_set_b['entailed_set_id']
						entailed_set_b_text = entailed_set_b['entailed_set'][0]['entailed_set_text']
						example = {
							'question_a_id': f'{question_id}|{answer_id}|{entailed_set_a_id}',
							'question_b_id': entailed_set_b_id,
							'question_a_text': entailed_set_a_text,
							'question_b_text': entailed_set_b_text,
						}
						self.examples.append(example)
			print(f'{question_id}: {len(entailed_sets)} entailed sets with {num_sets} total sets (possible {num_possible_sets})')

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		example = self.examples[idx]

		return example


class RGQEQuestionSelfPredictionDataset(Dataset):
	def __init__(self, input_path, search_path, queries, top_k):
		self.input_path = input_path
		self.search_path = search_path
		self.top_k = top_k
		self.queries = {q['question_id']: q for q in queries}

		with open(input_path) as f:
			self.answer_samples = json.load(f)

		question_answer_count = defaultdict(int)
		answer_questions = defaultdict(list)
		with open(self.search_path, 'r') as f:
			for line in f:
				line = line.strip()
				if line:
					question_id, _, answer_id, rank, score, run_name = line.split()
					if question_answer_count[question_id] > top_k:
						continue
					question_answer_count[question_id] += 1
					answer_questions[answer_id].append(question_id)

		self.examples = []
		for answer_id, question_ids in answer_questions.items():
			samples = self.answer_samples[answer_id]
			for question_id in question_ids:
				question_text = self.queries[question_id]['question']
				for s_id, sample_text in enumerate(samples):
					example = {
						'question_a_id': f'{answer_id}|{s_id}',
						'question_b_id': question_id,
						'question_a_text': sample_text,
						'question_b_text': question_text,
					}
					self.examples.append(example)

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		example = self.examples[idx]

		return example


class RGQETopQuestionPredictionDataset(Dataset):
	def __init__(self, input_path, search_path, top_k):
		self.input_path = input_path
		self.search_path = search_path
		self.top_k = top_k

		with open(input_path) as f:
			# [question_id][answer_id] -> entailed sets
			self.question_answer_sets = json.load(f)

		question_answer_count = defaultdict(int)
		question_samples = defaultdict(list)
		with open(self.search_path) as f:
			for line in f:
				line = line.strip()
				if line:
					question_id, _, answer_id, rank, score, run_name = line.split()
					if question_answer_count[question_id] >= top_k:
						continue
					question_answer_count[question_id] += 1
					if answer_id in self.question_answer_sets[question_id]:
						answer_sets = self.question_answer_sets[question_id][answer_id]
						for entailed_set in answer_sets:
							entailed_set_id = entailed_set['entailed_set_id']
							entailed_set_sample_text = entailed_set['entailed_set'][0]['sample_text']
							example = {
								'id': f'{question_id}|{answer_id}|{entailed_set_id}',
								'answer_id': answer_id,
								'entailed_set_text': entailed_set_sample_text
							}
							question_samples[question_id].append(example)

		self.examples = []
		for question_id, q_samples in question_samples.items():
			print(f'{question_id}: {len(q_samples)}')
			for sample_a, sample_b in itertools.combinations(q_samples, r=2):
				# ignore self entailment since that was already computed
				if sample_a['answer_id'] == sample_b['answer_id']:
					continue
				example = {
					'question_a_id': sample_a['id'],
					'question_b_id': sample_b['id'],
					'question_a_text': sample_a['entailed_set_text'],
					'question_b_text': sample_b['entailed_set_text'],
				}
				self.examples.append(example)

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		example = self.examples[idx]

		return example
