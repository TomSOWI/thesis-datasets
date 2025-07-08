import pandas as pd
from huggingface_hub import login
from datasets import load_dataset, DatasetDict, Dataset
import re
from tqdm import tqdm
tqdm.pandas()

login(token="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")


df = pd.read_parquet("/scratch/usr/nimtsspi/datasets/tg-focused-topic-dataset-280len-limit/df.parquet")
ds = Dataset.from_pandas(df)
ds.push_to_hub("TomData/tg-focused-topic-dataset-280len-limit", private = True)