#!/bin/bash

#SBATCH --job-name="r3_kd_t5_anli"
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=62G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --output=r_3minillm_kd.txt
#SBATCH --error=r_3minillm_err.txt


spack load cuda@11.4 || echo "Failed to load CUDA module" > /tmp/cuda_error_${SLURM_JOB_ID}.log
echo $CUDA_HOME



# Uncomment for smaller GPUs, reduce the split size to avoid OOM
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=128


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



BASE_PATH=${1-"/home/student/f/fklinge/share/bachelor/ichteste/minillm"}

#Add robust logging
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a job_progress.log
}
trap 'log "ERROR: Command failed with exit code $?"; exit 1' ERR




# First commands - write to a location you know is writable
echo "Job ${SLURM_JOB_ID} starting at $(date)" > /tmp/job_${SLURM_JOB_ID}_debug.log
echo "Working directory: $(pwd)" >> /tmp/job_${SLURM_JOB_ID}_debug.log
echo "User: $(whoami)" >> /tmp/job_${SLURM_JOB_ID}_debug.log
# Test if we can write to expected output locations
touch sft_anli_r1_enhanced.txt && echo "Can write to output file" >> /tmp/job_${SLURM_JOB_ID}_debug.log
touch err_sft_r1.txt && echo "Can write to error file" >> /tmp/job_${SLURM_JOB_ID}_debug.log


if [ ! -d "$BASE_PATH" ]; then
  echo "BASE_PATH directory doesn't exist: $BASE_PATH" > /tmp/job_error_${SLURM_JOB_ID}.log
  exit 1
fi


MASTER_ADDR=localhost
MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-1}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# models
CKPT_NAME="student"
TEACHER_CKPT_NAME="teacher"


if ! python ./fresh_models.py "$BASE_PATH" "$CKPT_NAME" "$TEACHER_CKPT_NAME" >> finetuning_out_test_loss.txt 2>&1; then
    echo "fresh_models.py failed. Exiting." >> finetuning_out_test_loss.txt
    exit 1
fi

CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}-minillm-r3/"
TEACHER_CKPT="${BASE_PATH}/checkpoints/${TEACHER_CKPT_NAME}-minillm/"

# data ## How to prepare data for t5, which collator is used in minillm
PROMPT_DATA_DIR="${BASE_PATH}/processed_data/anli_enhanced/full/r3/train/"
LM_DATA_DIR="${BASE_PATH}/processed_data/openwebtext/anli/full/r1/train/"
# runtime
SAVE_PATH="${BASE_PATH}/results/t5/train/minillm"
# hp
GRAD_ACC=6
BATCH_SIZE=12
# Probablyc chunks output length, rolling window, check importance later
CHUNK_SIZE=16
ANLI_ROUND="r3/"


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --n-nodes ${NNODES}"
OPTS+=" --fp32"
# OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --prompt-data-dir ${PROMPT_DATA_DIR}"
OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
OPTS+=" --dev-num 100"
OPTS+=" --num-workers 0"
OPTS+=" --anli-round ${ANLI_ROUND}"
# hp
OPTS+=" --epochs 1"
OPTS+=" --total-iters 100"
OPTS+=" --kd-ratio 0.5"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --lr 0.00007"
OPTS+=" --lr-min 5e-6"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --max-length 360"
OPTS+=" --max-prompt-length 180"
OPTS+=" --warmup-iters 100"
# runtime
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed 10"
OPTS+=" --seed-ppo 42"
OPTS+=" --seed-lm 7"
OPTS+=" --save-interval 1000"
OPTS+=" --eval-interval 500"
OPTS+=" --log-interval 50"
OPTS+=" --mid-log-num 1"
# ppo
OPTS+=" --type minillm"
OPTS+=" --ppo-epochs 2"
OPTS+=" --num-rollouts 32"
OPTS+=" --chunk-size ${CHUNK_SIZE}"
# minillm
OPTS+=" --length-norm"
#OPTS+=" --single-step-reg"
OPTS+=" --teacher-mixed-alpha 0.2"
# reward
OPTS+=" --reward-scaling 0.5"
OPTS+=" --cliprange-reward 100"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
# deepspeed
OPTS+=" --deepspeed"

OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_zero2.json"

export NCCL_DEBUG="INFO"
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >> debug.log
python -c "import torch; print('CUDA available:', torch.cuda.is_available())" >> debug.log
python -c "import os; print('CWD:', os.getcwd()); print('Files:', os.listdir('.'))" >> debug.log
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/train_minillm.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
