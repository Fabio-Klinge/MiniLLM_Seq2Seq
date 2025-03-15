#!/bin/sh

#SBATCH --job-name="data-preprocess-anli-t5-large"
#SBATCH --time=00:25:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --output=output.txt
#SBATCH --error=error.txt


#spack load cuda@11.4
#nvidia-smi

# Set --refinement 0

BASE_PATH=${1}
# Ensure there's a trailing slash on BASE_PATH or explicitly add slashes where needed
BASE_PATH=$(realpath "${BASE_PATH}")/

export TF_CPP_MIN_LOG_LEVEL=3
export HF_DATASETS_OFFLINE=1

# only prompt for MiniLLM train
# Small dataset > 0 defines length of dataset, else normal size
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_anli.py \
    --base-path ${BASE_PATH} \
    --data-dir ${BASE_PATH}data/anli/ \
    --processed-data-dir ${BASE_PATH}processed_data/anli_enhanced/prompt/ \
    --model-path ${BASE_PATH}checkpoints/flan-t5-large/ \
    --data-process-workers 8 \
    --max-prompt-length 320 \
    --dev-num 1000 \
    --small-dataset 0\
    --only-prompt \
    --model-type t5\
    --refinement 


# prompt and response for baselines
# Small dataset > 0 defines length of dataset, else normal size
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_anli.py \
    --base-path ${BASE_PATH} \
    --data-dir ${BASE_PATH}data/anli/ \
    --processed-data-dir ${BASE_PATH}processed_data/anli_enhanced/full/ \
    --model-path ${BASE_PATH}checkpoints/flan-t5-large/ \
    --data-process-workers 8 \
    --max-prompt-length 320 \
    --dev-num 1000 \
    --small-dataset 0\
    --model-type t5 \
    --refinement 
