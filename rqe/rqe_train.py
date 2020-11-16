
import sys
from transformers import BertTokenizer
import argparse
import os
import logging
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
# note: do NOT import torch before pytorch_lightning, really breaks TPUs
import torch

from rqe.model_utils import RQEBertFromSequenceClassification, RQEBertFromLanguageModel
from rqe.data_utils import BatchCollator, RQEDataset, load_clinical_data, load_quora_data, split_data


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', help='clinical-qe/quora', type=str, default='quora')
	parser.add_argument('--pre_model_name', type=str, default='nboost/pt-bert-base-uncased-msmarco')
	parser.add_argument('--pre_model_type', help='lm/seq', type=str, default='lm')
	parser.add_argument('--model_type', help='rqe', type=str, default='rqe')
	parser.add_argument('--seed', help='random seed', type=int, default=0)
	parser.add_argument('--batch_size', type=int, default=32)
	# 20 -> 10
	parser.add_argument('--epochs', type=int, default=10)
	# torch_cache_dir = '/users/max/data/models/torch_cache'
	parser.add_argument('--torch_cache_dir', type=str, default=None)
	# 5e-5 -> 5e-4
	parser.add_argument('--learning_rate', type=float, default=5e-4)
	parser.add_argument('--lr_warmup', type=float, default=0.1)
	parser.add_argument('--weight_decay', type=float, default=0.01)
	parser.add_argument('--save_directory', type=str, default='models')

	args = parser.parse_args()
	# TODO parameterize below into config file for reproducibility
	seed = args.seed
	pl.seed_everything(seed)
	dataset = args.dataset.lower()
	model_type = args.model_type.lower()
	pre_model_name = args.pre_model_name.lower()
	pre_model_type = args.pre_model_type.lower()

	torch_cache_dir = args.torch_cache_dir
	root_save_directory = args.save_directory

	batch_size = args.batch_size
	epochs = args.epochs
	learning_rate = args.learning_rate
	lr_warmup = args.lr_warmup
	weight_decay = args.weight_decay

	# TODO model_type
	if pre_model_type == 'seq':
		model_class = RQEBertFromSequenceClassification
	elif pre_model_type == 'lm':
		model_class = RQEBertFromLanguageModel
	else:
		raise ValueError(f'Unknown pre_model_type: {pre_model_type}')
	if dataset == 'clinical-qe':
		train_path = 'data/RQE_Data_AMIA2016/RQE_Train_8588_AMIA2016.xml'
		val_path = 'data/RQE_Data_AMIA2016/RQE_Test_302_pairs_AMIA2016.xml'
		# test_path = 'data/RQE_Data_AMIA2016/MEDIQA2019-Task2-RQE-TestSet-wLabels.xml'
		max_seq_len = 96
		# do 80% train 10% dev 10% test
		logging.info('Loading clinical-qe dataset...')
		train_examples = load_clinical_data(train_path)
		val_examples = load_clinical_data(val_path)

	elif dataset == 'quora':
		all_path = 'data/quora_duplicate_questions/quora_duplicate_questions.tsv'
		max_seq_len = 64
		# do 80% train 10% dev 10% test
		logging.info('Loading quora dataset...')
		examples = load_quora_data(all_path)
		# 80/20
		train_examples, other_examples = split_data(examples, ratio=0.8)
		# 10/10
		val_examples, _ = split_data(other_examples, ratio=0.5)

	else:
		raise ValueError(f'Unknown dataset: {dataset}')

	model_name = f'{dataset}-{model_type}-{pre_model_name.replace("/", "-")}'
	# export TPU_IP_ADDRESS=10.155.6.34
	# export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
	# TODO move to args
	use_tpus = True
	train_model = True
	load_model = False
	calc_seq_len = False
	is_distributed = False

	gpus = [0]
	precision = 16 if use_tpus else 32
	tpu_cores = 8
	num_workers = 4
	deterministic = True

	save_directory = os.path.join(root_save_directory, model_name)

	checkpoint_path = os.path.join(save_directory, 'pytorch_model.bin')

	if not os.path.exists(save_directory):
		os.mkdir(save_directory)

	# Also add the stream handler so that it logs on STD out as well
	# Ref: https://stackoverflow.com/a/46098711/4535284
	for handler in logging.root.handlers[:]:
		logging.root.removeHandler(handler)

	logfile = os.path.join(save_directory, "train_output.log")
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s [%(levelname)s] %(message)s",
		handlers=[
			logging.FileHandler(logfile, mode='w'),
			logging.StreamHandler()]
	)

	num_batches_per_step = (len(gpus) if not use_tpus else tpu_cores)
	updates_epoch = len(train_examples) // (batch_size * num_batches_per_step)
	updates_total = updates_epoch * epochs

	logging.info('Loading tokenizer...')
	tokenizer = BertTokenizer.from_pretrained(pre_model_name)
	train_dataset = RQEDataset(train_examples)
	val_dataset = RQEDataset(val_examples)

	train_data_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		collate_fn=BatchCollator(
			tokenizer,
			max_seq_len,
			force_max_seq_len=use_tpus
		)
	)

	if calc_seq_len:
		data_loader = DataLoader(
			train_dataset,
			batch_size=1,
			shuffle=True,
			num_workers=num_workers,
			collate_fn=BatchCollator(
				tokenizer,
				max_seq_len,
				force_max_seq_len=False
			)
		)
		import numpy as np
		from tqdm import tqdm
		logging.info('Calculating seq len stats...')
		seq_lens = []
		for batch in tqdm(data_loader):
			seq_len = batch['input_ids'].shape[-1]
			seq_lens.append(seq_len)
		p = np.percentile(seq_lens, 95)
		logging.info(f'95-percentile: {p}')
		exit()

	val_data_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		num_workers=num_workers,
		collate_fn=BatchCollator(
			tokenizer,
			max_seq_len,
			force_max_seq_len=use_tpus
		)
	)
	logging.info('Loading model...')
	model = model_class(
		pre_model_name=pre_model_name,
		learning_rate=learning_rate,
		lr_warmup=lr_warmup,
		updates_total=updates_total,
		weight_decay=weight_decay,
		torch_cache_dir=torch_cache_dir
	)
	if load_model:
		model.load_state_dict(torch.load(checkpoint_path))
	else:
		tokenizer.save_pretrained(save_directory)
		model.config.save_pretrained(save_directory)

	logger = pl_loggers.TensorBoardLogger(
		save_dir=save_directory,
		flush_secs=30,
		max_queue=2
	)

	if use_tpus:
		trainer = pl.Trainer(
			logger=logger,
			tpu_cores=tpu_cores,
			default_root_dir=save_directory,
			max_epochs=epochs,
			precision=precision,
			deterministic=deterministic
		)
	else:
		if len(gpus) > 1:
			backend = 'ddp' if is_distributed else 'dp'
		else:
			backend = None
		trainer = pl.Trainer(
			logger=logger,
			gpus=gpus,
			default_root_dir=save_directory,
			max_epochs=epochs,
			precision=precision,
			distributed_backend=backend,
			deterministic=deterministic
		)

	if train_model:
		logging.info('Training...')
		trainer.fit(model, train_data_loader, val_data_loader)
		logging.info('Saving checkpoint...')
		model.to('cpu')
		torch.save(model.state_dict(), checkpoint_path)

