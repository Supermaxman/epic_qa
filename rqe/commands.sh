#!/usr/bin/env bash

# v1
python -m rqe.rqe_train \
  --dataset=clinical-qe

python -m rqe.rqe_train \
  --dataset=quora
