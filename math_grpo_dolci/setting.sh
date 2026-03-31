#/usr/bin/env bash

pip install langdetect absl-py immutabledict nltk

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu129


python -c "import nltk; nltk.download('punkt_tab')"
