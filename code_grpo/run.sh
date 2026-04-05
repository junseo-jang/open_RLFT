#!/bin/bash
OUTPUT_DIR="./results"
LOG_DIR="./logs"
MODEL_DIR="./model/qwen3-1.7b-sft-by-tulu3-subsets"
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
NUM_GENERATIONS=8
BETA=0.04
WANDB_NAME="code-grpo"
SAVE_STEPS=100
CODE_API_URL="http://localhost:1234/test_program"
CODE_API_PORT=1234


export WANDB_PROJECT="Open_RLFT"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/..":$PYTHONPATH

if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

# Ctrl+C 시 vLLM 서버도 함께 종료
cleanup() {
    trap - SIGINT SIGTERM EXIT

    echo "Cleaning up..."
    # 학습 프로세스 정리 (run_grpo.py)
    pkill -9 -f "run_grpo.py" 2>/dev/null || true
    # vLLM 서버 정리
    if [ -n "$VLLM_PID" ]; then
        kill -9 $VLLM_PID 2>/dev/null || true
        pkill -9 -P $VLLM_PID 2>/dev/null || true
        wait $VLLM_PID 2>/dev/null || true
        VLLM_PID=""
    fi
    # 코드 실행 서버 정리
    if [ -n "$CODE_API_PID" ]; then
        kill -9 $CODE_API_PID 2>/dev/null || true
        wait $CODE_API_PID 2>/dev/null || true
        CODE_API_PID=""
    fi
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    fuser -k 8000/tcp 2>/dev/null || true
    fuser -k ${CODE_API_PORT}/tcp 2>/dev/null || true
    sleep 2
    echo "Cleanup done."
}
trap cleanup SIGINT SIGTERM EXIT

# 1. 코드 실행 API 서버를 백그라운드로 실행
uvicorn open_instruct.code_utils.api:app --host 0.0.0.0 --port $CODE_API_PORT &
CODE_API_PID=$!

echo "Code API server started at $(date)" | tee -a $LOG_FILE
echo "Waiting for Code API server..."
until curl -s http://localhost:${CODE_API_PORT}/health > /dev/null 2>&1; do
    sleep 2
done
echo "Code API server ready."

# 2. vLLM 서버를 백그라운드로 실행 (GPU 1장 할당)
CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
    --model $MODEL_DIR \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 &
VLLM_PID=$!

echo "vLLM server started at $(date)" | tee -a $LOG_FILE
echo "Waiting for vLLM server..."
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 10
done
echo "vLLM server ready."

# 3. GRPO 학습 실행 (나머지 GPU 7개)
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num_processes=7 run_grpo.py \
    --model_name $MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --vllm_mode server \
    --vllm_model_impl vllm \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --num_generations $NUM_GENERATIONS \
    --beta $BETA \
    --wandb_name $WANDB_NAME \
    --save_steps $SAVE_STEPS \
    --code_api_url $CODE_API_URL \
    --dataset_name allenai/Dolci-RL-Zero-Code-7B \
    2>&1 | tee -a $LOG_FILE

echo "Training completed at $(date)" | tee -a $LOG_FILE

# 4. 정리
cleanup
echo "All done at $(date)" | tee -a $LOG_FILE