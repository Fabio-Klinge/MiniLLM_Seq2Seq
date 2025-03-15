#!/bin/sh

#SBATCH --job-name="preprocess-opw"
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --output=opw.txt

BASE_PATH=${1}

MAX_LENGTH=256

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_openwebtext.py \
    --data-dir ${BASE_PATH}/data/openwebtext \
    --processed-data-dir ${BASE_PATH}/processed_data/openwebtext/flan-t5/${MAX_LENGTH}/ \
    --model-path ${BASE_PATH}/checkpoints/flan-t5-base\
    --max-length ${MAX_LENGTH} \
    --train-num 10000000 \
    --data-process-workers 32 \
    --dev-num 10000 \