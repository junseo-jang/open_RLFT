# Math GRPO Third

## Base Model

- **SFT Model**: `qwen3-1.7b-sft-by-tulu3-subsets`

## Training Data

- [allenai/RLVR-GSM](https://huggingface.co/datasets/allenai/RLVR-GSM) (train split)
- [allenai/RLVR-MATH](https://huggingface.co/datasets/allenai/RLVR-MATH) (train split)

두 데이터셋을 합친 뒤 shuffle(seed=42)하여 사용. 프롬프트 토큰 길이가 4096을 초과하는 샘플은 필터링.

## Training Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Algorithm | GRPO |
| Learning Rate | 1e-5 |
| Beta (KL penalty) | 0.04 |
| Num Epochs | 1 |
| Per Device Train Batch Size | 4 |
| Gradient Accumulation Steps | 4 |
| Effective Batch Size | 4 x 4 x 7 GPUs = 112 |
| Num Generations | 8 |
| Max Completion Length | 2048 |
| Save Strategy | steps (every 100 steps) |
| vLLM | server mode (GPU 1장), 학습 GPU 7장 |

## Evaluation

- **Generation max tokens**: 2048
- 자세한 벤치마크별 평가 결과는 [eval_results.md](eval_results.md) 참고
