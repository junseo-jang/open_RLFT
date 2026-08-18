# Open RLFT

Hugging Face [TRL](https://github.com/huggingface/trl)을 활용해 SFT(Supervised Fine-Tuning)부터 GRPO(Group Relative Policy Optimization) 기반 강화학습까지 전체 post-training 과정을 직접 학습하고 실험한 스터디 프로젝트입니다.

Qwen3-1.7B-Base를 instruction tuning한 뒤 instruction following·수학·코드 도메인별 reward를 적용해 GRPO로 학습하고, 각 단계에서 모델 성능이 어떻게 달라지는지 비교합니다. 분산 실행에는 [Accelerate](https://github.com/huggingface/accelerate), 생성 서버에는 [vLLM](https://github.com/vllm-project/vllm)을 사용하며 평가는 [NeMo-Skills](https://github.com/NVIDIA/NeMo-Skills)로 수행합니다.

## Study Goals

- TRL의 `SFTTrainer`를 사용해 base model을 instruction-tuned model로 학습하기
- TRL의 GRPO trainer 구조와 rollout, reward, KL penalty의 동작 이해하기
- Instruction following, math, code처럼 서로 다른 도메인의 reward function 구현하기
- SFT checkpoint와 GRPO checkpoint를 여러 벤치마크에서 비교하고 성능 간 trade-off 관찰하기

## Overview

```text
Qwen/Qwen3-1.7B-Base
        │
        ├── SFT: allenai/tulu-3-sft-mixture
        │
        └── GRPO
             ├── Instruction following: RLVR-IFeval + IF_multi_constraints
             ├── Math: RLVR-GSM + RLVR-MATH / Dolci-Instruct-RL
             └── Code: Dolci-RL-Zero-Code-7B
```

각 GRPO 실험은 SFT 모델 `seopbo/qwen3-1.7b-sft-by-tulu3-subsets`를 시작점으로 사용합니다.

## Experiments

| Experiment | Training data / reward | Documentation | Results |
|---|---|---|---|
| SFT | `allenai/tulu-3-sft-mixture` | [sft_trl](sft_trl/README.md) | [results](sft_trl/eval_results.md) |
| IFEval GRPO | `allenai/RLVR-IFeval`, `allenai/IF_multi_constraints_upto5` | [ifeval_grpo](ifeval_grpo/README.md) | [results](ifeval_grpo/eval_results.md) |
| Math GRPO | `allenai/RLVR-GSM`, `allenai/RLVR-MATH` | [math_grpo](math_grpo/README.md) | [results](math_grpo/eval_results.md) |
| Math GRPO (Dolci) | Math subsets of `allenai/Dolci-Instruct-RL` | [math_grpo_dolci](math_grpo_dolci/README.md) | [results](math_grpo_dolci/eval_results.md) |
| Math GRPO (third run) | `allenai/RLVR-GSM`, `allenai/RLVR-MATH` | [math_grpo_third](math_grpo_third/README.md) | [results](math_grpo_third/eval_results.md) |
| Code GRPO | `allenai/Dolci-RL-Zero-Code-7B` + execution reward | [code_grpo](code_grpo/README.md) | [results](code_grpo/eval_results.md) |

> `math_grpo`와 `math_grpo_dolci`의 결과 파일은 현재 비어 있으며, 실험 코드와 실행 설정만 포함되어 있습니다.

### Evaluation Results

- [SFT evaluation results](sft_trl/eval_results.md)
- [IFEval GRPO evaluation results](ifeval_grpo/eval_results.md)
- [Math GRPO evaluation results](math_grpo/eval_results.md)
- [Math GRPO (Dolci) evaluation results](math_grpo_dolci/eval_results.md)
- [Math GRPO third run evaluation results](math_grpo_third/eval_results.md)
- [Code GRPO evaluation results](code_grpo/eval_results.md)

## Repository Structure

```text
.
├── sft_trl/             # TRL SFTTrainer 기반 SFT 및 평가 결과
├── ifeval_grpo/         # Instruction-following reward 기반 GRPO
├── math_grpo/           # GSM8K + MATH 기반 GRPO
├── math_grpo_dolci/     # Dolci 수학 데이터 기반 GRPO
├── math_grpo_third/     # 수학 GRPO 추가 실험
├── code_grpo/           # 코드 실행 결과를 reward로 사용하는 GRPO
├── open_instruct/       # GRPO trainer, verifier, reward 관련 코드
├── lm_eval/             # vLLM + NeMo-Skills 평가 스크립트
└── upload_model_to_hf.* # Hugging Face Hub 업로드 도구
```

## Quick Start

### 1. Clone

```bash
git clone https://github.com/junseo-jang/open_RLFT.git
cd open_RLFT
```

### 2. Run SFT

```bash
cd sft_trl
pip install -r requirement.txt
bash sft_trl.sh
```

학습 기본값과 세부 설정은 [SFT documentation](sft_trl/README.md)을 참고하세요.

### 3. Run a GRPO experiment

예를 들어 Math GRPO는 다음과 같이 실행합니다.

```bash
cd math_grpo
bash setting.sh
python download.py
bash run.sh
```

각 실험 디렉터리는 공통적으로 다음 파일을 포함합니다.

- `setting.sh`: Python 패키지와 평가 의존성 설치
- `download.py`: SFT 모델을 로컬 `model/`에 다운로드
- `run_grpo.py`: 데이터 전처리, reward 함수, GRPO 설정
- `run.sh`: vLLM 서버와 분산 학습 프로세스 실행

현재 `run.sh` 설정은 A100 8장을 기준으로 GPU 0에서 vLLM을 실행하고 GPU 1–7에서 7개의 학습 프로세스를 구동합니다. GPU 수, 모델 경로, batch size 등은 실행 전에 각 스크립트 상단의 환경 변수와 인자를 환경에 맞게 수정해야 합니다.

Code GRPO는 모델 서버 외에 생성된 코드를 채점하는 로컬 execution API를 함께 실행합니다. IFEval GRPO는 Google IFEval 의존성과 verifier 설정이 추가로 필요하므로 각 실험 문서를 먼저 확인하세요.

## Evaluation

평가 스크립트는 IFEval, IFBench, GSM8K, Hendrycks MATH, Minerva MATH, HumanEval, MBPP를 지원하도록 구성되어 있습니다.

```bash
cd lm_eval
bash install_nemoskill.bash
bash launch_vllm.bash    # terminal 1
bash run_eval.bash       # terminal 2
```

`lm_eval/launch_vllm.bash`와 `lm_eval/run_eval.bash`에는 모델 및 출력 경로가 지정되어 있으므로 실행 환경에 맞게 먼저 변경하세요.

## Selected Results

모든 평가는 NeMo-Skills와 greedy decoding(`temperature=0`)을 사용했습니다. 아래 값은 서로 다른 실험 시점의 checkpoint에서 측정되었으므로 실험별 상세 표를 함께 확인해 주세요.

| Model / checkpoint | IFEval avg. | GSM8K | HumanEval base |
|---|---:|---:|---:|
| Initial SFT evaluation | 45.68 | 74.37 | 49.39 |
| IFEval GRPO (250 steps) | **71.36** | 10.08 | 46.34 |
| Math GRPO third (1,069 steps) | 54.34 | **84.15** | 59.76 |
| Code GRPO (950 steps) | 56.30 | 82.03 | 57.32 |

IFEval 특화 학습은 instruction-following 성능을 크게 높였지만 수학과 코드 벤치마크 성능을 낮췄습니다. 반면 Math GRPO third run은 GSM8K를 SFT 기준 82.71에서 84.15로 높였습니다. 자세한 checkpoint별 결과는 각 실험의 `eval_results.md`에서 확인할 수 있습니다.

## Notes

- 학습 스크립트는 Linux, CUDA, 다중 NVIDIA GPU 환경을 전제로 작성되었습니다.
- 일부 스크립트에는 `/root` 아래의 절대 경로가 포함되어 있습니다. 로컬 또는 클러스터 경로에 맞게 수정하세요.
- 모델 및 데이터셋 다운로드에는 Hugging Face 접근 권한과 충분한 디스크 공간이 필요합니다.
- 생성된 코드의 실행은 격리된 환경에서 수행하는 것을 권장합니다.
