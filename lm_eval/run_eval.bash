#!/bin/bash

# NeMo-Skills 벤치마크 평가 스크립트
# 벤치마크: ifeval, gsm8k, human-eval (순차 실행)
# Decoding: Greedy (temperature=0)

set -e

# ========== 설정 ==========
SERVER_TYPE="vllm"
MODEL_NAME="/root/project/sft_output/checkpoint-6250"
OUTPUT_DIR="/root/eval_results"
NEMO_SKILLS_DIR="/root/NeMo-Skills"

# vLLM은 API 키 검증을 하지 않지만, litellm 클라이언트가 요구함
export OPENAI_API_KEY="dummy"

# ========== 출력 디렉토리 생성 ==========
mkdir -p ${OUTPUT_DIR}
cd ${NEMO_SKILLS_DIR}

# ========== 벤치마크 (쉼표로 구분, 공백 사용 시 나머지가 Hydra로 넘어가 파싱 오류 발생) ==========
BENCHMARKS="ifeval"

ns eval \
    --server_type ${SERVER_TYPE} \
    --model ${MODEL_NAME} \
    --benchmarks ${BENCHMARKS} \
    --output_dir ${OUTPUT_DIR} \
    --expname greedy_eval \
    ++server.base_url=http://localhost:8000/v1 \
    ++max_concurrent_requests=8 \
    ++inference.tokens_to_generate=4096
