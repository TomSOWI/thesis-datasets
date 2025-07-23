import pandas as pd
from tqdm import tqdm
import re
from src.utils import to_parquet
from markdown_text_clean import clean_text
from tqdm import tqdm

# Relative import
import sys
import os
# Get the parent directory of the notebook (project root)
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from src.utils import show_examples, plot_distribution, investigate_patterns, ultimately_unescape


INPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"
OUTPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"
tqdm.pandas()


def main():
    print("=====Before processing=====")
    df = pd.read_parquet(f"{INPUT_PATH}/TG_280limit.parquet")
    # Check processing worked as intended
    investigate_patterns(df,text_column="message",n_examples=5)

    print("=====After processing=====")
    df = pd.read_parquet(f"{INPUT_PATH}/TG_unified.parquet")
    # Check processing worked as intended
    investigate_patterns(df,text_column="message",n_examples=5)
    

if __name__ == "__main__":
    main()