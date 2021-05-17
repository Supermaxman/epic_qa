#!/usr/bin/env bash

filename=$(basename -- "$0")
# run names
RUN_ID=${filename::-3}
RUN_NAME=HLTRI_COVID_RQE
DATASET=q_hier

RQE_PRE_MODEL_NAME=nboost/pt-bert-base-uncased-msmarco
RQE_MODEL_NAME=quora-seq-nboost-pt-bert-base-uncased-msmarco
RQE_MODEL_TYPE=seq
RQE_BATCH_SIZE=8
RQE_SAVE_DIRECTORY=models

RQE_NUM_GPUS=1

RQE_EVAL=true

export TOKENIZERS_PARALLELISM=true

echo "Starting experiment ${RUN_NAME}_${RUN_ID}"
echo "Reserving ${RQE_NUM_GPUS} GPU(s)..."
RQE_GPUS=$(python gpu/request_gpus.py -r ${RQE_NUM_GPUS})
if [[ ${RQE_GPUS} -eq -1 ]]; then
    echo "Unable to reserve ${RQE_NUM_GPUS} GPU(s), exiting."
    exit 1
fi
echo "Reserved ${RQE_NUM_GPUS} GPUs: ${RQE_GPUS}"

RQE_EVAL_GPUS=${RQE_GPUS}

handler()
{
    echo "Experiment aborted."
    echo "Freeing ${RQE_NUM_GPUS} GPUs: ${RQE_GPUS}"
    python gpu/free_gpus.py -i "${RQE_GPUS}"
    exit 1
}
trap handler SIGINT


if [[ ${RQE_EVAL} = true ]]; then
  python -m rqe.rqe_eval \
    --dataset=${DATASET} \
    --model_name="${RQE_MODEL_NAME}" \
    --pre_model_name=${RQE_PRE_MODEL_NAME} \
    --model_type=${RQE_MODEL_TYPE} \
    --batch_size=${RQE_BATCH_SIZE} \
    --save_directory=${RQE_SAVE_DIRECTORY} \
    --gpus "${RQE_EVAL_GPUS}"
fi


echo "Freeing ${RQE_NUM_GPUS} GPUs: ${RQE_GPUS}"
python gpu/free_gpus.py -i "${RQE_GPUS}"
