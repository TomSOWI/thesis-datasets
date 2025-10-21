#!/bin/bash

# default params
#SBATCH -p kisski                              # memory needed #old:40
#SBATCH -t 24:00:00                     # runtime upper limit
#SBATCH -G A100:1
#SBATCH -C 80gb_vram
#SBATCH --mem=240GB  
#SBATCH --job-name="pre-classify-big" 
#SBATCH --output=./logs/%x-%j.log       # logging output
#SBATCH -C inet                         # internet access 
#SBATCH --array=0-3


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
#HF + vLLm cache + wandb 
export HF_HOME=/mnt/vast-kisski/projects/kisski_tegami/hf_cache
export VLLM_CACHE_ROOT=/mnt/vast-kisski/projects/kisski_tegami/vllm_cache
export VLLM_WORKER_TMPDIR=/mnt/vast-kisski/projects/kisski_tegami/vllm_tmp
export TMPDIR=/mnt/vast-kisski/projects/kisski_tegami/vllm_tmp
export XDG_RUNTIME_DIR=/mnt/vast-kisski/projects/kisski_tegami/vllm_tmp
export WANDB_DIR=/mnt/vast-kisski/projects/kisski_tegami/wandb_cache

#Cuda for flash-atn
module load gcc/13.2.0
module load gcc/13.2.0-nvptx
module load cuda/12.6.2
export CUDA_HOME=/sw/rev/25.04/rome_mofed_cuda80_rocky8/linux-rocky8-zen2/gcc-13.2.0/cuda-12.6.2-i7i7owawqssmsubepstris6xo65ug4r7
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
#export TORCH_CUDA_ARCH_LIST="8.0"


# Arrays of models and task
MODELS=(
    magpie-babe-ft
    bert-base-uncased-hatexplain
    twitter-roberta-base-sentiment-latest
    bertweet-sexism
  
)
TASKS=(
    lexbias
    hate
    sentiment
    sexism
)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
export MODEL=$MODEL
export TASK=$TASK
echo "Running model=$MODEL for $Task"

python bias.py









