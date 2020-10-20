
from transformers import BertTokenizer
import argparse
import os
import logging
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from model_utils import QuestionAnsweringBert
from data_utils import SampleCollator, QuestionAnswerDataset, load_data, split_data
from sample_utils import UniformNegativeSampler


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', help='train/test', type=str, default='train')
	args = parser.parse_args()
	# TODO parameterize below into config file for reproducibility
	seed = 0
	mode = args.mode
	train_path = '/users/max/data/corpora/bioasq/2020/8b/training'
	test_path = '/users/max/data/corpora/bioasq/2020/8b/golden'
	save_directory = 'models'
	model_name = 'v1'
	pre_model_name = 'nboost/pt-biobert-base-msmarco'
	learning_rate = 5e-5
	lr_warmup = 0.1
	epochs = 10
	gradient_clip_val = 1.0
	weight_decay = 0.01
	max_seq_len = 512
	val_check_interval = 1.0
	is_distributed = True
	# export TPU_IP_ADDRESS=10.155.6.34
	# export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
	# batch_size = 64
	batch_size = 32
	negative_sample_size = 1
	accumulate_grad_batches = 1
	# accumulate_grad_batches = 4
	# gpus = [3, 4, 6, 7]
	gpus = [0]
	use_tpus = True
	precision = 16 if use_tpus else 32
	tpu_cores = 8
	num_workers = 1
	deterministic = True
	save_on_eval = False
	train_model = mode == 'train'
	load_model = mode != 'train'
	test_eval = mode == 'test'

	pl.seed_everything(seed)

	save_directory = os.path.join(save_directory, model_name)

	checkpoint_path = os.path.join(save_directory, 'model.ckpt')

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

	logging.info('Loading dataset...')
	examples, queries, answers = load_data(train_path)
	train_examples, val_examples = split_data(examples)

	num_batches_per_step = (len(gpus) if not use_tpus else tpu_cores)
	updates_epoch = len(train_examples) // (batch_size * num_batches_per_step)
	updates_total = updates_epoch * epochs

	callbacks = []
	logging.info('Loading collator...')

	train_neg_sampler = UniformNegativeSampler(
		answers,
		train_examples,
		negative_sample_size,
		seed=seed,
		train_callback=True
	)
	callbacks.append(train_neg_sampler)
	val_neg_sampler = UniformNegativeSampler(
		answers,
		train_examples,
		negative_sample_size,
		seed=seed,
		val_callback=True
	)
	callbacks.append(val_neg_sampler)

	tokenizer = BertTokenizer.from_pretrained(pre_model_name)
	train_dataset = QuestionAnswerDataset(train_examples)
	val_dataset = QuestionAnswerDataset(val_examples)

	# ensure negative_sample_size is correct based on batch_size
	train_collator = SampleCollator(
		tokenizer,
		train_neg_sampler,
		max_seq_len,
		force_max_seq_len=use_tpus
	)
	train_data_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		collate_fn=train_collator
	)

	val_collator = SampleCollator(
		tokenizer,
		val_neg_sampler,
		max_seq_len,
		force_max_seq_len=use_tpus
	)
	val_data_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		num_workers=num_workers,
		collate_fn=val_collator
	)

	logging.info('Loading model...')
	model = QuestionAnsweringBert(
		pre_model_name=pre_model_name,
		learning_rate=learning_rate,
		lr_warmup=lr_warmup,
		updates_total=updates_total,
		weight_decay=weight_decay
	)

	checkpoint_callback = None
	if save_on_eval:
		checkpoint_callback = ModelCheckpoint(
			verbose=False,
			monitor='val_loss',
			save_last=False,
			save_top_k=2,
			mode='min'
		)
	if use_tpus:
		logging.warning('Gradient clipping slows down TPU training drastically, disabled for now.')
		trainer = pl.Trainer(
			tpu_cores=tpu_cores,
			default_root_dir=save_directory,
			max_epochs=epochs,
			precision=precision,
			val_check_interval=val_check_interval,
			deterministic=deterministic,
			callbacks=callbacks,
			checkpoint_callback=checkpoint_callback,
			log_save_interval=50
		)
	else:
		if len(gpus) > 1:
			backend = 'ddp' if is_distributed else 'dp'
		else:
			backend = None
		trainer = pl.Trainer(
			gpus=gpus,
			default_root_dir=save_directory,
			max_epochs=epochs,
			precision=precision,
			val_check_interval=val_check_interval,
			distributed_backend=backend,
			gradient_clip_val=gradient_clip_val,
			deterministic=deterministic,
			callbacks=callbacks,
			checkpoint_callback=checkpoint_callback,
			log_save_interval=50
		)

	if train_model:
		logging.info('Training...')
		trainer.fit(model, train_data_loader, val_data_loader)
		logging.info('Saving checkpoint...')
		trainer.save_checkpoint(checkpoint_path)

	if test_eval:
		logging.info('Loading test data...')
		test_examples, queries, answers = load_data(test_path)
		# TODO
		# test_data_loader = DataLoader(
		# 	test_examples,
		# 	batch_size=batch_size,
		# 	shuffle=False,
		# 	drop_last=False,
		# 	num_workers=num_workers,
		# 	collate_fn=sampling.collate_fn_padding
		# )
		# logging.info('Computing thresholds on val data...')
		# trainer.test(model, val_data_loader)
		#
		# logging.info('Evaluating...')
		# trainer.test(model, test_data_loader)
