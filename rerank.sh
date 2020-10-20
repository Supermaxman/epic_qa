#!/usr/bin/env bash

python rerank_labels_sentence.py \
  --query_path data/consumer/consumer_questions_prelim.json \
  --collection_path data/consumer/version_2_split \
  --label_path data/consumer/consumer_fake_qrels.txt \
  --rerank_model models/consumer-v1 \
  --run_path runs/consumer/consumer-v1 \
  --batch_size 32 \
  --max_length 512 \
  --multi_sentence \
  --custom_model

python rerank_labels_sentence.py \
  --query_path data/expert/expert_questions_prelim.json \
  --collection_path data/expert/epic_qa_cord_2020-06-19_v2 \
  --label_path data/expert/qrels-covid_d4_j3.5-4.txt \
  --rerank_model models/expert-v1 \
  --run_path runs/expert/expert-v1 \
  --batch_size 32 \
  --max_length 512 \
  --multi_sentence \
  --custom_model