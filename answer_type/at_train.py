
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

from answer_type.model_utils import ATPBertFromLanguageModel
from answer_type.data_utils import BatchCollator, ATPDataset, load_smart_data, split_data


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', help='smart-dbpedia', type=str, default='smart-dbpedia')
	args = parser.parse_args()
	# TODO parameterize below into config file for reproducibility
	seed = 0
	pl.seed_everything(seed)
	dataset = args.dataset

	if dataset == 'smart-dbpedia':
		all_path = 'data/smart/smarttask_dbpedia_train.json'
		max_seq_len = 64
		# 32
		batch_size = 8
		pre_model_name = 'bert-base-uncased'
		model_class = ATPBertFromLanguageModel
		epochs = 20

		# do 80% train 10% dev 10% test
		logging.info('Loading smart dataset...')
		examples, category_map, types_map = load_smart_data(all_path)
		# 80/20
		train_examples, other_examples = split_data(examples, ratio=0.8)
		# 10/10
		val_examples, _ = split_data(other_examples, ratio=0.5)

	else:
		raise ValueError(f'Unknown dataset: {dataset}')

	save_directory = 'models'
	# torch_cache_dir = '/users/max/data/models/torch_cache'
	torch_cache_dir = None
	model_name = f'{dataset}-at-v1'
	learning_rate = 5e-5
	lr_warmup = 0.1
	gradient_clip_val = 1.0
	weight_decay = 0.01
	val_check_interval = 1.0
	is_distributed = False
	# export TPU_IP_ADDRESS=10.155.6.34
	# export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

	accumulate_grad_batches = 1
	# gpus = [3, 4, 6, 7]
	gpus = [0]
	use_tpus = False
	precision = 16 if use_tpus else 32
	# precision = 32
	tpu_cores = 8
	num_workers = 4
	deterministic = True
	train_model = True
	load_model = False

	calc_seq_len = False

	save_directory = os.path.join(save_directory, model_name)

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

	callbacks = []
	logging.info('Loading collator...')

	tokenizer = BertTokenizer.from_pretrained(pre_model_name)
	train_dataset = ATPDataset(train_examples)
	val_dataset = ATPDataset(val_examples)

	train_data_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		collate_fn=BatchCollator(
			tokenizer,
			max_seq_len,
			use_tpus,
			types_map
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
				False,
				types_map
			)
		)
		import numpy as np
		from tqdm import tqdm
		logging.info('Calculating seq len stats...')
		seq_lens = []
		max_id = -1
		for batch in tqdm(data_loader):
			seq_len = batch['input_ids'].shape[-1]
			max_id = max(max(batch['labels']), max_id)
			seq_lens.append(seq_len)
		p = np.percentile(seq_lens, 95)
		logging.info(f'95-percentile: {p}')
		logging.info(f'Max id: {max_id}')
		exit()

	val_data_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		num_workers=num_workers,
		collate_fn=BatchCollator(
			tokenizer,
			max_seq_len,
			use_tpus,
			types_map
		)
	)
	logging.info('Loading model...')
	model = model_class(
		pre_model_name=pre_model_name,
		learning_rate=learning_rate,
		lr_warmup=lr_warmup,
		updates_total=updates_total,
		weight_decay=weight_decay,
		category_map=category_map,
		types_map=types_map,
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
		logging.warning('Gradient clipping slows down TPU training drastically, disabled for now.')
		trainer = pl.Trainer(
			logger=logger,
			tpu_cores=tpu_cores,
			default_root_dir=save_directory,
			max_epochs=epochs,
			precision=precision,
			val_check_interval=val_check_interval,
			deterministic=deterministic,
			callbacks=callbacks
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
			val_check_interval=val_check_interval,
			distributed_backend=backend,
			gradient_clip_val=gradient_clip_val,
			deterministic=deterministic,
			callbacks=callbacks
		)

	if train_model:
		logging.info('Training...')
		trainer.fit(model, train_data_loader, val_data_loader)
		logging.info('Saving checkpoint...')
		model.to('cpu')
		torch.save(model.state_dict(), checkpoint_path)

