#!/usr/bin/env bash

pythoon rerank.py \
  --query_path data/epic_qa_prelim/consumer/consumer_questions_prelim.json \
  --collection_path data/epic_qa_prelim/consumer/version_2_split \
  --label_path data/epic_qa_prelim/prelim_judgments_corrected.json \
  --pre_model_name nboost/pt-biobert-base-msmarco \
  --model_name pt-biobert-base-msmarco




