# Evaluation Results (IFEval GRPO)

전체 데이터의 약 4%, 250 steps 학습 결과입니다.

평가 도구: [NeMo-Skills](https://github.com/NVIDIA/NeMo-Skills) | Decoding: Greedy (temperature=0)

## IFEval

| Metric | Before GRPO | After GRPO | Δ |
|--------|:-----------:|:----------:|:---:|
| Average Score | 55.38 | **71.36** | +15.98 |
| Prompt Strict Accuracy | 48.61 | 63.59 | +14.98 |
| Instruction Strict Accuracy | 59.47 | 73.02 | +13.55 |
| Prompt Loose Accuracy | 51.57 | 70.43 | +18.86 |
| Instruction Loose Accuracy | 61.87 | 78.42 | +16.55 |

## IFBench

| Metric | Before GRPO | After GRPO | Δ |
|--------|:-----------:|:----------:|:---:|
| Average Score | 13.86 | **28.73** | +14.87 |
| Prompt Strict Accuracy | 11.56 | 25.17 | +13.61 |
| Instruction Strict Accuracy | 12.84 | 25.07 | +12.23 |
| Prompt Loose Accuracy | - | 30.95 | - |
| Instruction Loose Accuracy | - | 33.73 | - |

## GSM8K

| Metric | Before GRPO | After GRPO | Δ |
|--------|:-----------:|:----------:|:---:|
| Symbolic Correct | 81.12 | 10.08 | -71.04 |
| No Answer | 2.88 | 0.00 | - |

## MATH (Hendrycks)

| Metric | Before GRPO | After GRPO | Δ |
|--------|:-----------:|:----------:|:---:|
| Symbolic Correct | 45.06 | 15.56 | -29.50 |
| No Answer | 45.44 | 1.60 | - |

## HumanEval

| Metric | Before GRPO | After GRPO | Δ |
|--------|:-----------:|:----------:|:---:|
| Passing Base Tests | 59.15 | 46.34 | -12.81 |
| Passing Plus Tests | 52.44 | 40.24 | -12.20 |

## MBPP

| Metric | Before GRPO | After GRPO | Δ |
|--------|:-----------:|:----------:|:---:|
| Passing Base Tests | 69.84 | 62.17 | -7.67 |
| Passing Plus Tests | 57.67 | 53.97 | -3.70 |
