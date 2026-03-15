#/usr/bin/env bash

pip install langdetect absl-py immutabledict nltk

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu129


# 임시로 받아서 open_instruct 폴더만 복사
git clone --filter=blob:none --sparse --depth=1 \
  https://github.com/allenai/open-instruct.git /tmp/open-instruct-tmp
cd /tmp/open-instruct-tmp
git sparse-checkout set open_instruct
git checkout

# open_instruct 디렉토리만 원하는 위치로 복사
cp -r open_instruct /root/open_instruct

# 임시 repo 삭제
rm -rf /tmp/open-instruct-tmp

# PYTHONPATH는 open_instruct의 "부모" 디렉토리를 넣어야 함
export PYTHONPATH=/root:$PYTHONPATH
echo 'export PYTHONPATH=/root:$PYTHONPATH' >> ~/.bashrc

python -c "import nltk; nltk.download('punkt_tab')"
