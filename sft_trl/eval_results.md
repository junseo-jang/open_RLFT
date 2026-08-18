# Evaluation Results

Qwen3-1.7B-Base SFT 모델의 벤치마크 평가 결과입니다.

평가 도구: [NeMo-Skills](https://github.com/NVIDIA/NeMo-Skills) | Decoding: Greedy (temperature=0)

## GSM8K

| Metric | Value |
|--------|-------|
| Symbolic Correct | **74.37** |
| No Answer | 0.83 |
| Num Entries | 1,319 |

## HumanEval

| Metric | Value |
|--------|-------|
| Passing Base Tests | **49.39** |
| Passing Plus Tests | **43.90** |
| Num Entries | 164 |

## IFEval

| Metric | Value |
|--------|-------|
| Average Score | **45.68** |
| Prompt Strict Accuracy | 34.75 |
| Instruction Strict Accuracy | 47.84 |
| Prompt Loose Accuracy | 43.44 |
| Instruction Loose Accuracy | 56.71 |
| Num Prompts | 541 |
