#/usr/bin/env bash

pip install git+https://github.com/seopbo/nemo-skills-patch.git
pip install langdetect absl-py immutabledict nltk

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu129


mkdir /opt/benchmarks
# https://github.com/seopbo/nemo-skills-patch/blob/main/dockerfiles/Dockerfile.nemo-skills
git clone https://github.com/google-research/google-research.git /opt/benchmarks/google-research --depth=1

python -c "import nltk; nltk.download('punkt_tab')"