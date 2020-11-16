#!/usr/bin/env bash

python -m rqe.rqe_train \
  --dataset=quora \
  --pre_model_name=bert-base-cased \
  --pre_model_type=lm

python -m rqe.rqe_train \
  --dataset=quora \
  --pre_model_name=bert-base-cased \
  --pre_model_type=lm \
  --learning_rate=5e-4 \
  --epochs=10

python -m rqe.rqe_train \
  --dataset=quora \
  --pre_model_name=bert-base-uncased \
  --pre_model_type=lm

python -m rqe.rqe_train \
  --dataset=quora \
  --pre_model_name=nboost/pt-bert-base-uncased-msmarco \
  --pre_model_type=seq

python -m rqe.rqe_train \
  --dataset=quora \
  --pre_model_name=nboost/pt-bert-large-msmarco \
  --pre_model_type=seq
