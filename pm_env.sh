#!/usr/bin/env bash

# :::::::::::::::::: Options ::::::::::::::::::
PYTHON_VERSION=$( bc <<< "3.8.10" )
ENV_NAME="edema_pm"
REQUIREMENTS_PATH="requirements/pm.txt"
# :::::::::::::::::::::::::::::::::::::::::::::

conda update -n base -c defaults conda --yes
conda create --name ${ENV_NAME} python=${PYTHON_VERSION} --no-default-packages --yes
conda init --all --dry-run --verbose
conda activate ${ENV_NAME}
python -V
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r ${REQUIREMENTS_PATH} --no-cache-dir
