import pandas as pd
from huggingface_hub import login
from datasets import load_dataset, DatasetDict, Dataset
import re
from tqdm import tqdm
tqdm.pandas()

login(token="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")

INPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"
OUTPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"

df = pd.read_parquet(f"{OUTPUT_PATH}/TG_target.parquet")
print(len(df.id.unique()))