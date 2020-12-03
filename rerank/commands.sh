#!/usr/bin/env bash

python -m rerank.rerank \
  --query_path data/epic_qa_prelim/consumer/consumer_questions_prelim.json \
  --collection_path data/epic_qa_prelim/consumer/version_2_split \
  --label_path data/epic_qa_prelim/prelim_judgments_corrected.json \
  --pre_model_name nboost/pt-biobert-base-msmarco \
  --model_name pt-biobert-base-msmarco

python -m rerank.format_preds \
  --run_path models/pt-biobert-base-msmarco/predictions.pt \
  --output_path models/pt-biobert-base-msmarco/HLTRI_RERANK_1.pred

python -m rerank.format_eval \
  --run_path models/pt-biobert-base-msmarco/HLTRI_RERANK_1.pred \
  --output_path models/pt-biobert-base-msmarco/HLTRI_RERANK_1.txt \

python rerank/epic_eval.py \
  data/epic_qa_prelim/prelim_judgments_corrected.json \
  models/pt-biobert-base-msmarco/HLTRI_RERANK_1.txt \
  rerank/.consumer_ideal_ranking_scores.tsv \
  --task consumer