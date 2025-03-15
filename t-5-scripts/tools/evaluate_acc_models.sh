#!/bin/bash

#SBATCH --job-name="Evaluate models"
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --output=eval_models.txt
#SBATCH --error=eval_models.txt

### ADJUST DS CONFIG TO MATCH HP ###

# After your SBATCH headers


BASE_PATH=${1-"/home/student/j/joldach/share"}

#!/bin/bash
#SBATCH directives...
#SLURM_JOB_ID=$("6662302")

# First commands - write to a location you know is writabl
# Add this check after BASE_PATH is defined



### DS CONFIG, bs, max_len, DATASET PATH, model paths




export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128



# model
MASTER_ADDR=localhost
MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
NNODES=1
NODE_RANK=0
# Adjust according to gpu request
GPUS_PER_NODE=${3-1}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

#R1
#CKPT_NAME="student:teacher:flan-t5-l-sft-r3:flan-t5-l-sft-r2-new:flan-t5-l-sft-r1-new"
CKPT_NAME="student-minillm-r1:student-minillm-r2:student-minillm-r3:"

# data
DATA_DIR="${BASE_PATH}/processed_data/anli_enhanced/full/"
ANLI_ROUND="r1:r2:r3"
# hp
# Batch size for each node

GRAD_ACC=6
EVAL_BATCH_SIZE=16
# length
MAX_LENGTH=350
# runtime
SAVE_PATH="${BASE_PATH}/results/model_evaluation/"
# seed
SEED=42


# model
OPTS+=" --base-path ${BASE_PATH}"


OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
#OPTS+=" --gradient-checkpointing"

# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --anli-round ${ANLI_ROUND}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num 1000"
#OPTS+=" --refinement"
# hp

OPTS+=" --batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 1500"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 0.01"
OPTS+=" --clip-grad 0.9"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 150"
OPTS+=" --min-prompt-length 20"
# runtime
#OPTS+=" --eval-gen"
OPTS+=" --save-interval 1000"
OPTS+=" --eval-interval 1000"
OPTS+=" --log-interval 25"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_eval.json"
# type
OPTS+=" --type lm"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 0.9"
OPTS+=" --temperature 0.7"
OPTS+=" --repetition-penalty 1.15"




export NCCL_DEBUG=INFO
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/evaluate_acc_models.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}