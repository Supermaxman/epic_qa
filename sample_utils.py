from abc import ABC, abstractmethod
import numpy as np
import torch
import pytorch_lightning as pl
import torch.distributed as dist
import os


class NegativeSampler(pl.Callback, ABC):
	def __init__(self):
		pass

	@abstractmethod
	def sample(self, pos_relation, batch_examples=None):
		pass


class UniformNegativeSampler(NegativeSampler):
	def __init__(
			self, answers, examples, negative_sample_size: int, seed=0, train_callback=False, val_callback=False,
			test_callback=False):
		super().__init__()
		self.negative_sample_size = negative_sample_size
		self.answers = {idx: answer for idx, answer in enumerate(answers)}
		self.examples = set([(ex['query']['id'], ex['answer']['id']) for ex in examples])
		self.seed = seed
		self.epoch = 0
		self.train_callback = train_callback
		self.val_callback = val_callback
		self.test_callback = test_callback
		self.gen = None
		self.gen_seed = None
		self.seen_seed = False

	def sample(self, pos_example, batch_examples=None):
		pos_query_id = pos_example['query']['id']
		num_samples = 0
		if not self.seen_seed:
			self.set_seed()
		while num_samples < self.negative_sample_size:
			sample_idx = torch.randint(low=0, high=len(self.answers), size=(1,), generator=self.gen)[0].item()
			sample_answer = self.answers[sample_idx]
			sample_answer_id = sample_answer['id']
			if (pos_query_id, sample_answer_id) not in self.examples:
				num_samples += 1
				sample_example = {
					'query': pos_example['query'],
					'answer': sample_answer
				}
				yield sample_example

	def set_seed(self):
		try:
			rank = dist.get_rank()
		except AssertionError:
			if 'XRT_SHARD_ORDINAL' in os.environ:
				rank = int(os.environ['XRT_SHARD_ORDINAL'])
			else:
				rank = 0
				print('No process group initialized, using default seed...')
		worker_info = torch.utils.data.get_worker_info()
		self.gen = torch.Generator()
		self.gen_seed = (self.seed, self.epoch, rank, worker_info.id)
		# print(f'Sampler using seed={self.gen_seed}')
		self.gen.manual_seed(hash(self.gen_seed))
		self.seen_seed = True

	def update_epoch(self, epoch):
		self.epoch = epoch
		self.seen_seed = False

	def on_train_epoch_start(self, trainer: pl.Trainer, pl_module):
		if self.train_callback:
			self.update_epoch(trainer.current_epoch)

	def on_validation_epoch_start(self, trainer, pl_module):
		if self.val_callback:
			self.update_epoch(epoch=0)

	def on_test_epoch_start(self, trainer, pl_module):
		if self.test_callback:
			self.update_epoch(epoch=0)
