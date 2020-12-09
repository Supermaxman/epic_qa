#!/usr/bin/env bash

export RUN_NAME=HLTRI_RERANK_RQE_24
export EXP_RUN_NAME=HLTRI_RERANK_RQE_21
export SEARCH_RUN=passage-large
export RERANK_RUN_NAME=HLTRI_RERANK_15
export DATASET=expert
export COLLECTION=epic_qa_prelim
export RGQE_THRESHOLD=0.8
export RQE_THRESHOLD=0.05

export RERANK_MODEL_NAME=rerank-expert-${SEARCH_RUN}-${RERANK_RUN_NAME}
export EXP_MODEL_NAME=docT5query-base
export RQE_MODEL_NAME=quora-seq-nboost-pt-bert-base-uncased-msmarco
export EXP_PRE_MODEL_NAME=models/docT5query-base/model.ckpt-1004000

export QUERY_PATH=data/${COLLECTION}/${DATASET}/questions.json
export LABEL_PATH=data/${COLLECTION}/${DATASET}/split/val.json
export COLLECTION_PATH=data/${COLLECTION}/${DATASET}/data
export SEARCH_PATH=models/${RERANK_MODEL_NAME}/${RERANK_RUN_NAME}.txt
export ANSWERS_PATH=models/${RERANK_MODEL_NAME}/${RERANK_RUN_NAME}.answers
export EXP_PATH=models/${EXP_MODEL_NAME}/${EXP_RUN_NAME}.exp
export RGQE_SELF_PATH=models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_self
export RGQE_CC_PATH=models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_cc
export RGQE_QUESTION_PATH=models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_question
export RGQE_TOP_PATH=models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_top
export RGQE_TOP_CC_PATH=models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_top_cc
export RGQE_ALL_PATH=models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_all
export RGQE_ALL_CC_SCORED_PATH=models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_all_cc_scored
export RUN_PATH=models/${RQE_MODEL_NAME}/${RUN_NAME}.txt
export EVAL_PATH=models/${RQE_MODEL_NAME}/${RUN_NAME}.eval

#python -m rerank.extract_answers \
#  --search_path ${SEARCH_PATH} \
#  --collection_path ${COLLECTION_PATH} \
#  --output_path ${ANSWERS_PATH}

# create expanded questions for every answer
#python expand_query./expand.py \
# --input_path ${SEARCH_PATH} \
# --collection_path ${COLLECTION_PATH} \
# --pre_model_name ${EXP_PRE_MODEL_NAME} \
# --model_name ${EXP_MODEL_NAME} \
# --top_k 20 \
# --num_samples 20 \
# --batch_size 16 \
#; \
#python -m expand_query.format_expand \
#  --model_path models/${EXP_MODEL_NAME} \
#  --output_path ${EXP_PATH}

# self entailment
python -m rgqe.rgqe \
  --input_path ${EXP_PATH} \
  --model_name models/${RQE_MODEL_NAME} \
  --max_seq_len 64 \
  --mode self \
; \
python -m rgqe.format_rgqe_self \
  --model_path models/${RQE_MODEL_NAME} \
  --output_path ${RGQE_SELF_PATH} \
; \
python -m rgqe.rgqe_self_components \
  --input_path ${RGQE_SELF_PATH} \
  --expand_path ${EXP_PATH} \
  --output_path ${RGQE_CC_PATH} \
  --threshold ${RGQE_THRESHOLD}

# query-question entailment to filter out bad generated questions
python -m rgqe.rgqe \
  --input_path ${RGQE_CC_PATH} \
  --search_path ${SEARCH_PATH} \
  --query_path ${QUERY_PATH} \
  --label_path ${LABEL_PATH} \
  --model_name models/${RQE_MODEL_NAME} \
  --max_seq_len 64 \
  --mode question \
; \
python -m rgqe.format_rgqe_question \
  --model_path models/${RQE_MODEL_NAME} \
  --output_path ${RGQE_QUESTION_PATH}

# top_k set entailment
python -m rgqe.rgqe \
  --input_path ${RGQE_CC_PATH} \
  --qe_path ${RGQE_QUESTION_PATH} \
  --search_path ${SEARCH_PATH} \
  --model_name models/${RQE_MODEL_NAME} \
  --max_seq_len 64 \
  --mode top \
  --top_k 100 \
  --threshold ${RQE_THRESHOLD} \
; \
python -m rgqe.format_rgqe_top \
  --model_path models/${RQE_MODEL_NAME} \
  --output_path ${RGQE_TOP_PATH} \
; \
python -m rgqe.rgqe_top_components \
  --input_path ${RGQE_TOP_PATH} \
  --cc_path ${RGQE_CC_PATH} \
  --output_path ${RGQE_TOP_CC_PATH} \
  --threshold ${RGQE_THRESHOLD}

# all entailment against sets
python -m rgqe.rgqe \
  --input_path ${RGQE_TOP_CC_PATH} \
  --cc_path ${RGQE_CC_PATH} \
  --qe_path ${RGQE_QUESTION_PATH} \
  --model_name models/${RQE_MODEL_NAME} \
  --max_seq_len 64 \
  --mode all \
; \
python -m rgqe.format_rgqe_all \
  --model_path models/${RQE_MODEL_NAME} \
  --output_path ${RGQE_ALL_PATH}

python -m rgqe.rgqe_all_components \
  --input_path ${RGQE_ALL_PATH} \
  --cc_path ${RGQE_TOP_CC_PATH} \
  --answers_path ${ANSWERS_PATH} \
  --queries_path ${QUERY_PATH} \
  --output_path ${RGQE_ALL_CC_SCORED_PATH} \
  --threshold ${RGQE_THRESHOLD} \
  --ratio 0.8 \
; \
python -m rgqe.format_eval \
  --results_path ${RGQE_ALL_CC_SCORED_PATH} \
  --output_path ${RUN_PATH} \
; \
python rerank/epic_eval.py \
  ${LABEL_PATH} \
  ${RUN_PATH} \
  rerank/.${DATASET}_ideal_ranking_scores.tsv \
  --task ${DATASET} \
  | tail -n 3 \
  | awk \
    '{ for (i=1; i<=NF; i++) RtoC[i]= (RtoC[i]? RtoC[i] FS $i: $i) }
    END{ for (i in RtoC) print RtoC[i] }' \
  | tail -n 2 > ${EVAL_PATH} \
; \
cat ${EVAL_PATH}


