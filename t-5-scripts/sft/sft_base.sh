#!/bin/bash

#SBATCH --job-name="r1_sft_t5_xl_anli"
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=240G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --output=sft_anli_xl_r1_enhanced.txt
#SBATCH --error=err_sft_xl_r1.txt

### ADJUST DS CONFIG TO MATCH HP ###

# After your SBATCH headers
log_memory() {
  echo "====== MEMORY STATUS $(date) ======" >> memory_log.txt
  free -h >> memory_log.txt
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv >> memory_log.txt
  echo "" >> memory_log.txt
}

# Add periodic logging
(while true; do log_memory; sleep 30; done) &
MEMORY_PID=$!
trap "kill $MEMORY_PID" EXIT


BASE_PATH=${1-"/home/student/j/joldach/share"}
CKPT_NAME="flan-t5-xl"

#!/bin/bash
#SBATCH directives...
#SLURM_JOB_ID=$("6662302")

# First commands - write to a location you know is writable
echo "Job ${SLURM_JOB_ID} starting at $(date)" > /tmp/job_${SLURM_JOB_ID}_debug.log
echo "Working directory: $(pwd)" >> /tmp/job_${SLURM_JOB_ID}_debug.log
echo "User: $(whoami)" >> /tmp/job_${SLURM_JOB_ID}_debug.log
# Test if we can write to expected output locations
touch sft_anli_r1_enhanced.txt && echo "Can write to output file" >> /tmp/job_${SLURM_JOB_ID}_debug.log
touch err_sft_r1.txt && echo "Can write to error file" >> /tmp/job_${SLURM_JOB_ID}_debug.log

# Add this check after BASE_PATH is defined
if [ ! -d "$BASE_PATH" ]; then
  echo "BASE_PATH directory doesn't exist: $BASE_PATH" > /tmp/job_error_${SLURM_JOB_ID}.log
  exit 1
fi



if ! python ./fresh_model.py "$BASE_PATH" "$CKPT_NAME" >> finetuning_out_test_loss.txt 2>&1; then
    echo "fresh_model.py failed. Exiting." >> finetuning_out_test_loss.txt
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONUNBUFFERED=1

# Add robust logging
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a job_progress.log
}
trap 'log "ERROR: Command failed with exit code $?"; exit 1' ERR


spack load cuda@11.4 || echo "Failed to load CUDA module" > /tmp/cuda_error_${SLURM_JOB_ID}.log
nvidia-smi


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



# Model name instead of path downloads the model
CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}-new/"
# data
DATA_DIR="${BASE_PATH}/processed_data/anli_enhanced/full/"
ANLI_ROUND="r1/"
# hp
# Batch size for each node
BATCH_SIZE=2
LR=0.00007
GRAD_ACC=8
EVAL_BATCH_SIZE=2
# length
MAX_LENGTH=350
# runtime
SAVE_PATH="${BASE_PATH}/results/${CKPT_NAME}/train/sft"
# seed
SEED=42


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --model-type ${CKPT_NAME}"
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
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 1500"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 0.01"
OPTS+=" --clip-grad 0.9"
OPTS+=" --epochs 5"
OPTS+=" --loss-ratio 0.8"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 100"
OPTS+=" --min-prompt-length 20"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
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
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_sft.json"
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
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune_t5.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}