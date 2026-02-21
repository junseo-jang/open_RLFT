#!/bin/bash

# NeMo-Skills 설치 스크립트

set -e  # 에러 발생 시 중단

echo "=== blinker 충돌 해결 및 nemo-evaluator-launcher 설치 ==="
pip install --ignore-installed blinker
pip install nemo-evaluator-launcher

echo "=== NeMo-Skills GitHub에서 클론 ==="
cd /root

git clone https://github.com/NVIDIA/NeMo-Skills.git


echo "=== NeMo-Skills 설치 ==="
cd NeMo-Skills
pip install -e .

python /root/NeMo-Skills/nemo_skills/dataset/ifeval/prepare.py
python /root/NeMo-Skills/nemo_skills/dataset/gsm8k/prepare.py
python /root/NeMo-Skills/nemo_skills/dataset/human-eval/prepare.py

echo "=== ifeval 평가 의존성 설치 ==="
mkdir -p /opt/benchmarks
cd /opt/benchmarks
if [ ! -d "google-research" ]; then
    git clone --depth 1 https://github.com/google-research/google-research.git
    echo "google-research 설치 완료"
else
    echo "google-research 이미 설치됨"
fi

echo "=== ifeval 추가 패키지 설치 ==="
pip install langdetect immutabledict absl-py nltk

echo "=== NLTK 데이터 다운로드 ==="
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"

echo "=== 설치 완료! ==="
ns --help

