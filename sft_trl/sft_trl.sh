#!/bin/bash

MODEL_NAME="Qwen/Qwen3-1.7B-Base"
DATASET_NAME="allenai/tulu-3-sft-mixture"

OUTPUT_DIR="./sft_output"
LOG_DIR="./logs"
NUM_TRAIN_EPOCHS=1
WARMUP_STEPS=None
WARMUP_RATIO=0.1
LOGGING_STEPS=10
SAVE_STEPS=250
SAVE_TOTAL_LIMIT=2
WANDB_PROJECT="Open_RLFT"
RUN_NAME=""
export WANDB_PROJECT=${WANDB_PROJECT}
RESUME_FROM_CHECKPOINT="False"

PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=32
LEARNING_RATE=2e-5

LEARNING_RATE_SCHEDULER_TYPE="cosine"

MAX_LENGTH=4096
CHAT_TEMPLATE_PATH="./chat_template.jinja"

# Create log directory
mkdir -p $LOG_DIR

# Generate log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

echo "Log file: $LOG_FILE"
echo "Starting training at $(date)" | tee -a $LOG_FILE

NUM_GPU=$(python -c "import torch; print(torch.cuda.device_count())")
accelerate launch --num_processes=$NUM_GPU ./sft_trl.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio $WARMUP_RATIO \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --bf16 \
    --lr_scheduler_type $LEARNING_RATE_SCHEDULER_TYPE \
    --max_length $MAX_LENGTH \
    --assistant_only_loss \
    --chat_template_path $CHAT_TEMPLATE_PATH \
    --model_dtype bfloat16 \
    --run_name $RUN_NAME \
    --resume_from_checkpoint $RESUME_FROM_CHECKPOINT \
    2>&1 | tee -a $LOG_FILE

echo "Training finished at $(date)" | tee -a $LOG_FILE