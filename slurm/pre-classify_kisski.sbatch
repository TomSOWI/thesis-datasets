#!/bin/bash

# default params
#SBATCH -p kisski         
#SBATCH -t 01:00:00                     # runtime upper limit
#SBATCH -G A100:4                    # GPU_name:amount 
#SBATCH --mem=40GB                      # memory needed #old:40
#SBATCH --job-name="test" 
#SBATCH --output=./logs/%x-%j.log       # logging output
#SBATCH -C inet                         # internet access

# notification
#SBATCH --mail-type=all
#SBATCH --mail-user=tomklotz@gmx.net                # receiving email address

# Project root dir
PROJECT_PATH=/user/spinde/u11191/thesis-datasets/telegram/pre-classify
CWD=$(pwd)
cd "$PROJECT_PATH" || exit 1

# Micromamba setup
export PATH="/mnt/vast-kisski/projects/kisski_tegami/micromamba/bin:$PATH"
eval "$(/mnt/vast-kisski/projects/kisski_tegami/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate main

# Environment variables
#HF + vLLm cache
export HF_HOME=/mnt/vast-kisski/projects/kisski_tegami/hf_cache
#export VLLM_CACHE_ROOT=/mnt/vast-kisski/projects/kisski_tegami/vllm_cache


# To be used in code script
export INPUT_PATH=/mnt/vast-kisski/projects/kisski_tegami
export OUTPUT_PATH=/mnt/vast-kisski/projects/kisski_tegami/datasets
export OFFLINE=false # If set too false add publisher names to model


# Run python script
python main.py