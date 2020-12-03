import json
import argparse
import os
import pytorch_lightning as pl
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from rerank.model_utils import RerankBert
from rerank.data_utils import QueryPassageDataset, PredictionBatchCollator
import logging
from pytorch_lightning import loggers as pl_loggers


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-q', '--query_path', required=True)
	parser.add_argument('-c', '--collection_path', required=True)
	parser.add_argument('-r', '--run_path', required=True)
	parser.add_argument('-s', '--search_run', default=None)
	parser.add_argument('-l', '--label_path', default='data/prelim_judgments.json')
	parser.add_argument('-pm', '--pre_model_name', default='nboost/pt-biobert-base-msmarco')
	parser.add_argument('-mn', '--model_name', default='pt-biobert-base-msmarco')
	parser.add_argument('-sd', '--save_directory', default='models')
	parser.add_argument('-bs', '--batch_size', default=64, type=int)
	parser.add_argument('-ml', '--max_seq_len', default=128, type=int)
	parser.add_argument('-ms', '--multi_sentence', default=False, action='store_true')
	parser.add_argument('-ng', '--n_gram_max', default=3, type=int)
	parser.add_argument('-se', '--seed', default=0, type=int)
	parser.add_argument('-cd', '--torch_cache_dir', default=None)

	args = parser.parse_args()
	seed = args.seed
	pl.seed_everything(seed)

	save_directory = args.save_directory
	model_name = args.model_name
	save_directory = os.path.join(save_directory, model_name)

	if not os.path.exists(save_directory):
		os.mkdir(save_directory)

	collection_path = args.collection_path
	label_path = args.label_path
	search_run = args.search_run
	query_path = args.query_path
	pre_model_name = args.pre_model_name
	tokenizer_name = pre_model_name
	batch_size = args.batch_size
	max_seq_len = args.max_seq_len
	multi_sentence = args.multi_sentence
	n_gram_max = args.n_gram_max
	torch_cache_dir = args.torch_cache_dir

	is_distributed = False
	# export TPU_IP_ADDRESS=10.155.6.34
	# export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
	gpus = [0]
	use_tpus = True
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

	# TODO allow for use of search qrels
	# qrels = defaultdict(list)
	# with open(search_run, 'r') as f:
	# 	for line in f:
	# 		line = line.strip().split()
	# 		if line:
	# 			query_id, _, doc_id, dq_rank = line
	# 			query_id = int(query_id)
	# 			dq_rank = int(dq_rank)
	# 			if dq_rank > 0:
	# 				qrels[query_id].append(doc_id)

	logging.info(f'Loading tokenizer: {tokenizer_name}')
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
	logging.info(f'Loading dataset: {collection_path}')
	eval_dataset = QueryPassageDataset(
		collection_path,
		queries,
		multi_sentence,
		n_gram_max
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
	model = RerankBert(
		pre_model_name=pre_model_name,
		learning_rate=5e-5,
		lr_warmup=0.1,
		updates_total=0,
		weight_decay=0.01,
		torch_cache_dir=torch_cache_dir,
		predict_mode=True
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

# print(f'Running reranking on passages and writing results to {run_path}...')
	# pass_scores = defaultdict(list)
	# for query_id, query in tqdm(enumerate(queries, start=1), total=len(queries)):
	#
	# 	with torch.no_grad():
	# 		# score_layer = torch.nn.Softmax(dim=-1)
	# 		for batch in tqdm(dataloader, total=len(dataloader)):
	# 			logits = model(
	# 				input_ids=batch['input_ids'].to(device),
	# 				token_type_ids=batch['token_type_ids'].to(device),
	# 				attention_mask=batch['attention_mask'].to(device)
	# 			)[0]
	# 			# scores = score_layer(logits)
	# 			# positive probability
	# 			scores = logits[:, 1].cpu().numpy()
	#
	# 			# make larger score mean better answer
	# 			pass_scores[query_id].extend(zip(batch['id'], scores))
	#
	# 	# sort large to small
	# 	pass_scores[query_id] = list(sorted(pass_scores[query_id], key=lambda x: x[1], reverse=True))
	#
	#
	# print(f'Saving results...')
	# with open(run_path, 'w') as fo:
	# 	for query_id, passages in pass_scores.items():
	# 		for rank, (doc_pass_sent_id, score) in enumerate(passages, start=1):
	# 			line = f'{query_id}\tQ0\t{doc_pass_sent_id}\t{rank}\t{score:.8f}\t{run_name}\n'
	# 			fo.write(line)
	# print('Done!')