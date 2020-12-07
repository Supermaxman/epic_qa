
import json
import argparse
import os
from collections import defaultdict
import pytorch_lightning as pl
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from rerank.model_utils import RerankBert
from rerank.data_utils import QueryPassageLabeledDataset, RerankBatchCollator
import logging
from pytorch_lightning import loggers as pl_loggers
import torch


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-q', '--query_path', required=True)
	parser.add_argument('-c', '--collection_path', required=True)
	parser.add_argument('-ps', '--passage_search_run', default=None)
	parser.add_argument('-ds', '--document_search_run', default=None)
	parser.add_argument('-s', '--split_path', default=None)
	parser.add_argument('-l', '--label_path', default='data/prelim_judgments.json')
	parser.add_argument('-pm', '--pre_model_name', default='nboost/pt-biobert-base-msmarco')
	parser.add_argument('-mn', '--model_name', default='pt-biobert-base-msmarco')
	parser.add_argument('-sd', '--save_directory', default='models')
	parser.add_argument('-bs', '--batch_size', default=16, type=int)
	parser.add_argument('-ml', '--max_seq_len', default=128, type=int)
	parser.add_argument('-ms', '--multi_sentence', default=False, action='store_true')
	parser.add_argument('-ng', '--n_gram_max', default=3, type=int)
	parser.add_argument('-se', '--seed', default=0, type=int)
	parser.add_argument('-cd', '--torch_cache_dir', default=None)
	parser.add_argument('-cs', '--calc_seq_len', default=False, action='store_true')
	parser.add_argument('-tpu', '--use_tpus', default=False, action='store_true')
	parser.add_argument('-opa', '--only_passages', default=False, action='store_true')

	args = parser.parse_args()
	seed = args.seed
	pl.seed_everything(seed)

	save_directory = args.save_directory
	model_name = args.model_name
	save_directory = os.path.join(save_directory, model_name)
	checkpoint_path = os.path.join(save_directory, 'pytorch_model.bin')

	if not os.path.exists(save_directory):
		os.mkdir(save_directory)

	collection_path = args.collection_path
	label_path = args.label_path
	split_path = args.split_path
	passage_search_run = args.passage_search_run
	document_search_run = args.document_search_run
	query_path = args.query_path
	pre_model_name = args.pre_model_name
	tokenizer_name = pre_model_name
	batch_size = args.batch_size
	max_seq_len = args.max_seq_len
	multi_sentence = args.multi_sentence
	n_gram_max = args.n_gram_max
	torch_cache_dir = args.torch_cache_dir
	calc_seq_len = args.calc_seq_len
	only_passages = args.only_passages

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

	epochs = 10
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
	query_id_to_question_id = {}
	query_id = 1
	with open(query_path) as f:
		all_queries = json.load(f)
		for query in all_queries:
			query_id_to_question_id[query_id] = query['question_id']
			query_id += 1
			if keep_ids is not None and query['question_id'] not in keep_ids:
				continue
			else:
				queries.append(query)

	document_qrels = None
	if document_search_run is not None:
		document_qrels = defaultdict(set)
		with open(document_search_run, 'r') as f:
			for line in f:
				line = line.strip().split()
				if line:
					query_id, _, doc_id, dq_rank = line
					query_id = int(query_id)
					question_id = query_id_to_question_id[query_id]
					dq_rank = int(dq_rank)
					if dq_rank > 0:
						document_qrels[question_id].add(doc_id)

	passage_qrels = None
	if passage_search_run is not None:
		passage_qrels = defaultdict(set)
		with open(passage_search_run, 'r') as f:
			for line in f:
				line = line.strip().split()
				if line:
					question_id, _, doc_pass_id, dq_rank, dq_score, dq_run = line
					passage_qrels[question_id].add(doc_pass_id)

	logging.info(f'Loading tokenizer: {tokenizer_name}')
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
	logging.info(f'Loading dataset: {collection_path}')

	with open(os.path.join(split_path, 'train.json'), 'r') as f:
		train_labels = json.load(f)

	with open(os.path.join(split_path, 'val.json'), 'r') as f:
		val_labels = json.load(f)

	train_dataset = QueryPassageLabeledDataset(
		collection_path,
		queries,
		train_labels,
		multi_sentence,
		n_gram_max,
		document_qrels,
		passage_qrels,
		only_passages
	)
	val_dataset = QueryPassageLabeledDataset(
		collection_path,
		queries,
		val_labels,
		multi_sentence,
		n_gram_max,
		document_qrels,
		passage_qrels,
		only_passages
	)

	train_data_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		collate_fn=RerankBatchCollator(
			tokenizer,
			max_seq_len,
			use_tpus
		)
	)
	val_data_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		collate_fn=RerankBatchCollator(
			tokenizer,
			max_seq_len,
			use_tpus
		)
	)

	num_batches_per_step = (len(gpus) if not use_tpus else tpu_cores)
	updates_epoch = len(train_dataset) // (batch_size * num_batches_per_step)
	updates_total = updates_epoch * epochs
	logging.info('Loading model...')
	model = RerankBert(
		pre_model_name=pre_model_name,
		learning_rate=5e-5,
		lr_warmup=0.1,
		updates_total=updates_total,
		weight_decay=0.01,
		torch_cache_dir=torch_cache_dir
	)

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
			max_epochs=epochs,
			precision=precision,
			distributed_backend=backend,
			gradient_clip_val=1.0,
			deterministic=deterministic
		)

	logging.info('Training...')
	trainer.fit(model, train_data_loader, val_data_loader)
	logging.info('Saving checkpoint...')
	model.to('cpu')
	torch.save(model, checkpoint_path)

