#!/bin/bash
#SBATCH --job-name=political_leaning
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --mail-user=tomklotz@gmx.net
#SBATCH --output=telegram/pre-classify/logs/%x-%j.log


# Load required modules
#module unload sw.clx.el9
#module load cuda sw.a100

# Move to project directory
cd /home/nimtsspi/thesis-datasets/telegram/pre-classify|| exit 1

# Initialize micromamba shell and activate environment
eval "$(micromamba shell hook -s bash)"
micromamba activate tk_ma

# Optional debugging info
# micromamba list
# which python
# python --version
# python -m torch.utils.collect_env
# nvidia-smi

# Run your script
#python gender.py
#python hatespeech.py
#python lexbias.py
#python sentiment.py
#python politicalness.py
#python political_leaning.py

