import sys
import json
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
from answer_type.data_utils import PredictionBatchCollator, ATPDataset, load_quora_data, load_smart_maps

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', help='smart-dbpedia', type=str, default='smart-dbpedia')
	args = parser.parse_args()
	# TODO parameterize below into config file for reproducibility
	seed = 0
	pl.seed_everything(seed)
	dataset = args.dataset

	save_directory = 'models'
	model_name = f'{dataset}-at-v3'
	save_directory = os.path.join(save_directory, model_name)

	checkpoint_path = os.path.join(save_directory, 'pytorch_model.bin')

	if not os.path.exists(save_directory):
		os.mkdir(save_directory)

	category_map_path = os.path.join(save_directory, 'category_map.json')
	types_map_path = os.path.join(save_directory, 'types_map.json')

	if dataset == 'smart-dbpedia':
		all_path = 'data/quora_duplicate_questions/quora_duplicate_questions.tsv'
		max_seq_len = 64
		# 32
		batch_size = 128
		pre_model_name = 'bert-base-uncased'
		model_class = ATPBertFromLanguageModel
		epochs = 50

		logging.info('Loading quora dataset...')
		category_map, types_map = load_smart_maps(category_map_path, types_map_path)
		eval_examples = load_quora_data(all_path)[:1000]
	else:
		raise ValueError(f'Unknown dataset: {dataset}')

	torch_cache_dir = '/users/max/data/models/torch_cache'
	# torch_cache_dir = None
	# learning_rate = 5e-5
	learning_rate = 5e-4
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

	callbacks = []
	logging.info('Loading collator...')

	tokenizer = BertTokenizer.from_pretrained(pre_model_name)
	eval_dataset = ATPDataset(eval_examples)

	eval_data_loader = DataLoader(
		eval_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		collate_fn=PredictionBatchCollator(
			tokenizer,
			max_seq_len,
			use_tpus
		)
	)

	logging.info('Loading model...')
	model = model_class(
		pre_model_name=pre_model_name,
		learning_rate=learning_rate,
		lr_warmup=lr_warmup,
		updates_total=0,
		weight_decay=weight_decay,
		category_map=category_map,
		types_map=types_map,
		torch_cache_dir=torch_cache_dir,
		predict_mode=True
	)
	model.load_state_dict(torch.load(checkpoint_path))

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

	logging.info('Evaluating...')
	try:
		trainer.test(model, eval_data_loader)
	except Exception as e:
		logging.exception('Exception during evaluating', exc_info=e)


