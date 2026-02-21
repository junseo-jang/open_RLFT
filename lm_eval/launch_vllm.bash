python -m vllm.entrypoints.openai.api_server \
    --model /root/project/sft_output/checkpoint-6250 \
    --port 8000 \
    --dtype auto \
    --gpu-memory-utilization 0.9