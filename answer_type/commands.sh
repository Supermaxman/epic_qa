#!/usr/bin/env bash

# v1
python -m answer_type.at_train \
  --dataset=smart-dbpedia

python -m answer_type.at_train \
  --dataset=quora
