#!/usr/bin/env bash

export RUN_NAME=HLTRI_RERANK_RQE_21
export SEARCH_RUN=passage-large
export RERANK_RUN_NAME=HLTRI_RERANK_15
export RERANK_MODEL_NAME=rerank-expert-${SEARCH_RUN}-${RERANK_RUN_NAME}
export EXP_MODEL_NAME=docT5query-base
export RQE_MODEL_NAME=quora-seq-nboost-pt-bert-base-uncased-msmarco
export EXP_PRE_MODEL_NAME=models/docT5query-base/model.ckpt-1004000
export DATASET=expert
export COLLECTION=epic_qa_prelim

# create expanded questions for every answer
python -m expand_query.expand \
 --input_path models/${RERANK_MODEL_NAME}/${RERANK_RUN_NAME}.txt \
 --collection_path data/${COLLECTION}/${DATASET}/data \
 --pre_model_name ${EXP_PRE_MODEL_NAME} \
 --model_name ${EXP_MODEL_NAME} \
 --top_k 20 \
 --num_samples 20 \
 --batch_size 16 \
; \
python -m expand_query.format_expand \
  --model_path models/${EXP_MODEL_NAME} \
  --output_path models/${EXP_MODEL_NAME}/${RUN_NAME}.exp \

# self entailment
python -m rgqe.rgqe \
  --input_path models/${EXP_MODEL_NAME}/${RUN_NAME}.exp \
  --model_name models/${RQE_MODEL_NAME} \
  --mode self \
; \
python -m rgqe.format_rgqe_self \
  --model_path models/${RQE_MODEL_NAME} \
  --output_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_self \
; \
python -m rgqe.rgqe_self_components \
  --input_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_self \
  --expand_path models/${EXP_MODEL_NAME}/${RUN_NAME}.exp \
  --output_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_cc \
  --threshold 0.5

python -m rgqe.rgqe \
  --input_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_cc \
  --search_path models/${RERANK_MODEL_NAME}/${RERANK_RUN_NAME}.txt \
  --query_path data/${COLLECTION}/${DATASET}/questions.json \
  --label_path data/${COLLECTION}/${DATASET}/split/val.json \
  --model_name models/${RQE_MODEL_NAME} \
  --mode question \
  --top_k 100 \
; \
python -m rgqe.format_rgqe_question \
  --model_path models/${RQE_MODEL_NAME} \
  --output_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_question


# top_k set entailment
python -m rgqe.rgqe \
  --input_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_cc \
  --search_path models/${RERANK_MODEL_NAME}/${RERANK_RUN_NAME}.txt \
  --model_name models/${RQE_MODEL_NAME} \
  --mode top \
  --top_k 100 \
; \
python -m rgqe.format_rgqe_top \
  --model_path models/${RQE_MODEL_NAME} \
  --output_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_top \
; \
python -m rgqe.rgqe_top_components \
  --input_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_top \
  --expand_path models/${EXP_MODEL_NAME}/${RUN_NAME}.exp \
  --output_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_top_cc \
  --threshold 0.5



#python -m expand_query.format_run \
#  --input_path models/${EXP_MODEL_NAME}/${RUN_NAME}.exp \
#  --scores_path models/${RERANK_MODEL_NAME}/${RERANK_RUN_NAME}.txt \
#  --output_path models/${RQE_MODEL_NAME}/${RUN_NAME}.exp_scored

#
python -m rqe.rqe \
  --input_path models/${RERANK_MODEL_NAME}/${RERANK_RUN_NAME}.txt \
  --expand_path models/${EXP_MODEL_NAME}/${RUN_NAME}.exp \
  --query_path data/${COLLECTION}/${DATASET}/questions.json \
  --label_path data/${COLLECTION}/${DATASET}/split/val.json \
  --model_name models/${RQE_MODEL_NAME} \
; \
python -m rqe.format_rqe \
  --model_path models/${RQE_MODEL_NAME} \
  --expand_path models/${EXP_MODEL_NAME}/${RUN_NAME}.exp \
  --output_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rqe \
  --threshold 0.01 \
  --num_samples 1 \
; \
python -m rqe.format_run \
  --rqe_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rqe \
  --scores_path models/${RERANK_MODEL_NAME}/${RERANK_RUN_NAME}.txt \
  --output_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rqe_scored

python -m rgqe.rgqe \
  --input_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rqe_scored \
  --model_name models/${RQE_MODEL_NAME} \
  --mode all \
; \
python -m rgqe.format_rgqe \
  --model_path models/${RQE_MODEL_NAME} \
  --output_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe \
  --threshold 0.5

python -m rgqe.format_eval \
  --results_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rqe_scored \
  --rgqe_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe \
  --output_path models/${RQE_MODEL_NAME}/${RUN_NAME}_RGQE.txt \
  --threshold 0.5 \
  --overlap_ratio 1.0 \
  --overall_ratio 0.0 \
; \
python rerank/epic_eval.py \
  data/${COLLECTION}/${DATASET}/split/val.json \
  models/${RQE_MODEL_NAME}/${RUN_NAME}.txt \
  rerank/.${DATASET}_ideal_ranking_scores.tsv \
  --task ${DATASET} \
  | tail -n 3 \
  | awk \
    '{ for (i=1; i<=NF; i++) RtoC[i]= (RtoC[i]? RtoC[i] FS $i: $i) }
    END{ for (i in RtoC) print RtoC[i] }' \
  | tail -n 2 > models/${RQE_MODEL_NAME}/${RUN_NAME}.eval \
; \
cat models/${RQE_MODEL_NAME}/${RUN_NAME}.eval
