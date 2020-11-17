#!/usr/bin/env bash

python -m rqe.rqe_train \
  --dataset=quora \
  --pre_model_name=bert-base-cased \
  --model_type=lm

python -m rqe.rqe_train \
  --dataset=quora \
  --pre_model_name=bert-base-uncased \
  --model_type=lm

python -m rqe.rqe_train \
  --dataset=quora \
  --pre_model_name=nboost/pt-bert-base-uncased-msmarco \
  --model_type=seq

python -m rqe.rqe_train \
  --dataset=quora \
  --pre_model_name=sentence-transformers/bert-base-nli-mean-tokens \
  --model_type=lm

# version_3 has info concatenated with cls
# version_ has attention pooling for each category + concat
python -m rqe.rqe_train \
  --dataset=quora \
  --pre_model_name=nboost/pt-bert-base-uncased-msmarco \
  --model_type=seq-at