# SFT with TRL

Qwen3-1.7B-Base 모델을 [tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) 데이터셋으로 SFT(Supervised Fine-Tuning) 학습하고, NeMo-Skills 기반 벤치마크로 평가하는 실험입니다.

## Files

```
sft_trl/
├── sft_trl.py          # 학습 스크립트 (TRL SFTTrainer)
├── sft_trl.sh          # 학습 실행 쉘 스크립트
├── chat_template.jinja # 채팅 템플릿
└── requirement.txt     # Python 의존성
```

## Training

### 1. 의존성 설치

```bash
cd sft_trl
pip install -r requirement.txt
```

### 2. 학습 실행

```bash
bash sft_trl.sh
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `Qwen/Qwen3-1.7B-Base` |
| Dataset | `allenai/tulu-3-sft-mixture` |
| Epochs | 1 |
| Batch Size | 1 (per device) |
| Gradient Accumulation | 32 |
| Learning Rate | 2e-5 |
| LR Scheduler | Cosine |
| Max Length | 4096 |
| Precision | bfloat16 |

## Evaluation

[NeMo-Skills](https://github.com/NVIDIA/NeMo-Skills)를 사용하여 GSM8K, HumanEval, IFEval 벤치마크로 평가합니다.

```
lm_eval/
├── install_nemoskill.bash  # NeMo-Skills 설치
├── launch_vllm.bash        # vLLM 서버 실행
└── run_eval.bash           # 벤치마크 평가 실행
```

### 1. NeMo-Skills 설치

```bash
cd lm_eval
bash install_nemoskill.bash
```

### 2. vLLM 서버 실행

```bash
bash launch_vllm.bash
```

### 3. 벤치마크 평가 실행

```bash
bash run_eval.bash
```

## Evaluation Results

자세한 평가 결과는 [eval_results.md](eval_results.md)를 참고하세요.

| Benchmark | Metric | Score |
|-----------|--------|-------|
| GSM8K | Symbolic Correct | 74.37 |
| HumanEval | Passing Base Tests | 49.39 |
| HumanEval | Passing Plus Tests | 43.90 |
| IFEval | Average Score | 45.68 |
