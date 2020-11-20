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
# version_13 has attention pooling for each category + concat
python -m rqe.rqe_train \
  --dataset=quora \
  --pre_model_name=nboost/pt-bert-base-uncased-msmarco \
  --model_type=seq-at

# version_
python -m rqe.rqe_train \
  --dataset=quora \
  --pre_model_name=nboost/pt-bert-base-uncased-msmarco \
  --model_type=seq-at

# 0.8894
python -m rqe.rqe_eval \
  --dataset=quora \
  --pre_model_name=bert-base-uncased \
  --model_type=lm

# 0.8955
python -m rqe.rqe_eval \
  --dataset=quora \
  --pre_model_name=nboost/pt-bert-base-uncased-msmarco \
  --model_type=seq

# 0.8947
python -m rqe.rqe_eval \
  --dataset=quora \
  --pre_model_name=nboost/pt-bert-base-uncased-msmarco \
  --model_type=seq-at


# baselines from https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3119-4
# Logistic Regression + Features    0.6779
# NN                                0.8134
# NN + GloVe embeddings             0.8362
# # # # # # # # # # # # # # # #
# bert-base                         0.8894
# bert-base-msmarco                 0.8955
