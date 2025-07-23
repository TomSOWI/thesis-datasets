#!/bin/bash
#SBATCH --job-name=tg-preprocess
#SBATCH --partition=cpu-clx:test
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=1:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --mail-user=tomklotz@gmx.net
#SBATCH --output=logs/%x-%j.log



# Load required modules
#module unload sw.clx.el9
#module load cuda sw.a100

# Move to project directory
cd /home/nimtsspi/thesis-datasets/telegram|| exit 1

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
#python combine_batches.py --requires internet
python explore_character_len.py
python unify_messages.py
python topic_filter.py

