# Open RLFT

Qwen3-1.7B-Base 모델을 대상으로 SFT 및 GRPO 기반 강화학습 실험을 진행하는 프로젝트입니다.

## Project Structure

```
.
├── sft_trl/                # SFT 학습 코드 (TRL SFTTrainer)
├── ifeval_grpo/            # IFEval 기반 GRPO 학습 코드
└── lm_eval/                # 모델 평가 코드 (NeMo-Skills)
```

## Experiments

| Experiment | Description | Details |
|------------|-------------|---------|
| SFT | tulu-3-sft-mixture로 Supervised Fine-Tuning | [sft_trl/README.md](sft_trl/README.md) |
| IFEval GRPO | IFEval 보상 기반 GRPO 강화학습 | [ifeval_grpo/README.md](ifeval_grpo/README.md) |
