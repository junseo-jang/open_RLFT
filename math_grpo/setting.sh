#/usr/bin/env bash

echo "=== blinker 충돌 해결 및 nemo-evaluator-launcher 설치 ==="
pip install --ignore-installed blinker
pip install nemo-evaluator-launcher

echo "=== NeMo-Skills GitHub에서 클론 ==="
cd /root

git clone https://github.com/NVIDIA/NeMo-Skills.git


echo "=== NeMo-Skills 설치 ==="
cd NeMo-Skills
pip install -e .

pip install -r /root/open_RLFT/math_grpo/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu129
