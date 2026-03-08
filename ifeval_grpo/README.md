# IFEval GRPO

SFT로 학습된 Qwen3-1.7B 모델을 IFEval 기반 보상 함수로 GRPO(Group Relative Policy Optimization) 강화학습하는 실험입니다.

## Files

```
ifeval_grpo/
├── setting.sh              # 환경 구성 스크립트
├── download.py             # 베이스 모델 다운로드
├── my_utils.py             # IFEvalVerifier 유틸리티 (open_instruct/로 이동 필요)
├── run.sh                  # 학습 실행 스크립트
├── run_grpo.py             # GRPO 학습 메인 코드
└── requirements.txt        # Python 의존성
```

## Setup

### 1. 환경 구성

```bash
bash setting.sh
```

다음 작업이 자동으로 수행됩니다:
- Python 패키지 설치 (`requirements.txt`)
- [open-instruct](https://github.com/allenai/open-instruct)에서 `open_instruct/` 디렉토리를 `/root/open_instruct`로 복사
- `PYTHONPATH=/root` 환경 변수 설정
- IFEval 평가용 google-research 레포 클론 (`/opt/benchmarks/google-research`)

### 2. 모델 다운로드

```bash
python download.py
```

SFT로 학습된 베이스 모델(`seopbo/qwen3-1.7b-sft-by-tulu3-subsets`)을 `./model/` 경로에 다운로드합니다.

### 3. my_utils.py 이동

```bash
cp my_utils.py /root/open_instruct/my_utils.py
```

`run_grpo.py`는 `from open_instruct.my_utils import IFEvalVerifier`로 유틸리티를 참조하므로, `my_utils.py`를 `open_instruct/` 디렉토리 아래로 이동해야 합니다.

## Training

### 학습 실행

```bash
bash run.sh
```

A100 8장 기준으로, GPU 1장은 vLLM 서버, 나머지 7장은 GRPO 학습에 사용합니다.

```
GPU 0       → vLLM 서버 (generation)
GPU 1~7     → GRPO 학습 (accelerate, 7 processes)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `seopbo/qwen3-1.7b-sft-by-tulu3-subsets` |
| Dataset | `allenai/RLVR-IFeval` + `allenai/IF_multi_constraints_upto5` |
| Learning Rate | 1e-5 |
| Per Device Batch Size | 4 |
| Gradient Accumulation | 4 |
| Num Generations | 8 |
| Beta (KL penalty) | 0.01 |
| Max Completion Length | 8192 |
| GPU | A100 × 8 (vLLM 1 + training 7) |

### Reward Function

두 데이터셋의 `ground_truth` 포맷에 따라 자동 분기하는 통합 보상 함수를 사용합니다:

- `RLVR-IFeval`: `func_name` 기반 JSON → Google IFEval instruction_id로 변환 후 검증
- `IF_multi_constraints`: `IFEvalVerifier`가 직접 처리

## Datasets

| Dataset | Description |
|---------|-------------|
| [allenai/RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval) | IFEval 형식의 instruction-following RL 데이터 |
| [allenai/IF_multi_constraints_upto5](https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5) | 최대 5개 복합 제약 조건 데이터 |

## Evaluation Results

전체 데이터의 약 4%, 250 steps 학습 결과입니다. 자세한 점수는 [eval_results.md](eval_results.md)를 참고

Train Wandb : https://wandb.ai/jjs97612/Open_RLFT?nw=nwuserjjs97612

| Benchmark | Before GRPO | After GRPO | Δ |
|-----------|:-----------:|:----------:|:---:|
| IFEval (avg) | 55.38 | **71.36** | +15.98 |
| IFBench (avg) | 13.86 | **28.73** | +14.87 |
| GSM8K | 81.12 | 10.08 | -71.04 |
| MATH | 45.06 | 15.56 | -29.50 |
| HumanEval | 59.15 | 46.34 | -12.81 |
| MBPP | 69.84 | 62.17 | -7.67 |
