import json
import argparse
import os
from collections import defaultdict
import pytorch_lightning as pl
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from rqe.model_utils import RQEPredictionBert
from rqe.data_utils import RQEPredictionDataset, PredictionBatchCollator
import logging
from pytorch_lightning import loggers as pl_loggers
import torch


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-e', '--expand_path', required=True)
	parser.add_argument('-q', '--query_path', required=True)
	parser.add_argument('-l', '--label_path', default='data/prelim_judgments.json')
	parser.add_argument('-mn', '--model_name', default='models/quora-seq-at-nboost-pt-bert-base-uncased-msmarco')
	parser.add_argument('-pm', '--pre_model_name', default='nboost/pt-bert-base-uncased-msmarco')
	parser.add_argument('-bs', '--batch_size', default=32, type=int)
	parser.add_argument('-ml', '--max_seq_len', default=64, type=int)
	parser.add_argument('-se', '--seed', default=0, type=int)
	parser.add_argument('-cd', '--torch_cache_dir', default=None)
	parser.add_argument('-tpu', '--use_tpus', default=False, action='store_true')

	args = parser.parse_args()
	seed = args.seed
	pl.seed_everything(seed)

	model_name = args.model_name
	checkpoint_path = os.path.join(model_name, 'pytorch_model.bin')
	pre_model_name = args.pre_model_name
	save_directory = model_name

	input_path = args.input_path
	expand_path = args.expand_path
	query_path = args.query_path
	label_path = args.label_path
	tokenizer_name = model_name
	batch_size = args.batch_size
	max_seq_len = args.max_seq_len
	torch_cache_dir = args.torch_cache_dir

	is_distributed = False
	# export TPU_IP_ADDRESS=10.155.6.34
	# export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
	gpus = [0]
	use_tpus = args.use_tpus
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

	logging.info('Loading queries...')
	keep_ids = None
	if label_path:
		keep_ids = set()
		with open(label_path) as f:
			questions = json.load(f)
			for question in questions:
				question_id = question['question_id']
				keep_ids.add(question_id)

	queries = []
	with open(query_path) as f:
		all_queries = json.load(f)
		for query in all_queries:
			if keep_ids is not None and query['question_id'] not in keep_ids:
				continue
			else:
				queries.append(query)

	logging.info(f'Loading tokenizer: {tokenizer_name}')
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
	logging.info(f'Loading dataset')

	eval_dataset = RQEPredictionDataset(
		expand_path,
		input_path,
		queries
	)
	eval_data_loader = DataLoader(
		eval_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		collate_fn=PredictionBatchCollator(
			tokenizer,
			max_seq_len,
			use_tpus
		)
	)

	logging.info('Loading model...')
	model = RQEPredictionBert(
		pre_model_name=pre_model_name,
		learning_rate=5e-5,
		lr_warmup=0.1,
		updates_total=0,
		weight_decay=0.01,
		torch_cache_dir=torch_cache_dir,
		predict_mode=True,
		predict_path=save_directory
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
			max_epochs=0,
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
			max_epochs=0,
			precision=precision,
			distributed_backend=backend,
			gradient_clip_val=1.0,
			deterministic=deterministic
		)

	logging.info('Evaluating...')
	try:
		trainer.test(model, eval_data_loader)
	except Exception as e:
		logging.exception('Exception during evaluating', exc_info=e)

