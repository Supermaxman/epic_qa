
import os
import argparse
import pytorch_lightning as pl
import logging
from transformers import T5Tokenizer
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader
from expand_query.data_utils import AnswerDataset, PredictionBatchCollator
from expand_query.model_utils import T5QueryGenerator


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--collection_path', required=True)
  parser.add_argument('-ps', '--input_path', required=True)
  parser.add_argument('-pm', '--pre_model_name', default='docT5query-base/model.ckpt-1004000')
  parser.add_argument('-tn', '--tokenizer_name', default='t5-base')
  parser.add_argument('-mn', '--model_name', default='docT5query-base')
  parser.add_argument('-sd', '--save_directory', default='models')
  parser.add_argument('-bs', '--batch_size', default=32, type=int)
  parser.add_argument('-ml', '--max_seq_len', default=96, type=int)
  parser.add_argument('-se', '--seed', default=0, type=int)
  parser.add_argument('-cd', '--torch_cache_dir', default=None)
  parser.add_argument('-mo', '--max_output_length', default=64, type=int)
  parser.add_argument('-k', '--top_k', default=10, type=int)
  parser.add_argument('-s', '--num_samples', default=10, type=int)

  args = parser.parse_args()
  seed = args.seed
  pl.seed_everything(seed)
  save_directory = args.save_directory
  model_name = args.model_name
  save_directory = os.path.join(save_directory, model_name)

  if not os.path.exists(save_directory):
    os.mkdir(save_directory)

  collection_path = args.collection_path
  input_path = args.input_path
  pre_model_name = args.pre_model_name
  tokenizer_name = args.tokenizer_name
  batch_size = args.batch_size
  max_seq_len = args.max_seq_len
  torch_cache_dir = args.torch_cache_dir

  max_output_length = args.max_output_length
  top_k = args.top_k
  num_samples = args.num_samples

  is_distributed = False
  # export TPU_IP_ADDRESS=10.155.6.34
  # export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
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

  logging.info(f'Loading tokenizer: {tokenizer_name}')
  tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
  logging.info(f'Loading dataset: {collection_path}')
  eval_dataset = AnswerDataset(
    collection_path,
    input_path
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
  model = T5QueryGenerator(
    pre_model_name=pre_model_name,
    learning_rate=5e-5,
    lr_warmup=0.1,
    updates_total=0,
    weight_decay=0.01,
    max_output_length=max_output_length,
    top_k=top_k,
    num_samples=num_samples,
    tokenizer=tokenizer,
    torch_cache_dir=torch_cache_dir,
    predict_mode=True,
    predict_path=save_directory
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
