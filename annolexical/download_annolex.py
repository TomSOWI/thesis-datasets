from datasets import load_dataset
import pandas as pd
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

INPUT_PATH = "mediabiasgroup/anno-lexical"  # HF
OUTPUT_PATH = "/scratch/usr/nimtsspi/datasets/annolex"

login(token="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")
ds = load_dataset(INPUT_PATH)

df1 = ds["train"].to_pandas()
df1["split"] = "train"

df2 = ds["validation"].to_pandas()
df2["split"] = "validation"

df3 = ds["test"].to_pandas()
df3["split"] = "test"

df = pd.concat([df1, df2, df3], ignore_index=True)
print(df.columns)
df.to_parquet(f"{OUTPUT_PATH}/base.parquet")
