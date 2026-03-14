#/usr/bin/env bash

pip install git+https://github.com/seopbo/nemo-skills-patch.git

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu129
