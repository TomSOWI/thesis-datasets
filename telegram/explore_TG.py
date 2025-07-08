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
topic_mapping = pd.read_csv(f"{INPUT_PATH}/ch_to_topic_mapping.csv")
df = pd.merge(df, topic_mapping, how="left", left_on="id", right_on="ch_ID")
df = df.drop("ch_ID", axis=1)

def _unify_text(text):
    """Unify text e.g. remove URLs, etc.
    """
    
    if text != text:
        return text

    #text = unicodedata.normalize('NFKD', text) # "déjà vu" → "deja vu"

    #text = re.sub(r"#[A-Za-z0-9_]+", "", text)  # remove #hashtag
    text = re.sub(r"RT\ ", " ", text)  # remove 'RT' from tweets
    text = re.sub(r"@[A-Za-z0-9_]+", "@user", text)  # remove @user

    text = re.sub(r'\*\*',"",text) # "remove bold asterisks"
    text =  re.sub(r'_{2,}', '_', text) # remove multiple "_" 
    
    text = re.sub(r'\[.*?\]\(https?://[^\)]+\)', '', text) # remove attachement links
    text = re.sub(r'https?://\S+', '', text) # remove remaining links
    text = re.sub(r"https?://[A-Za-z0-9./]+", " ", text)  # remove links
    text = re.sub("\t", " ", text)  # remove tab
    text = re.sub("\n", " ", text)  # remove newlines
    text = re.sub("\r", " ", text)  # remove \r type newlines
    text = re.sub(r" +", " ", text)  # remove multiple whitespaces
    text = re.sub(r"linebreak", "", text)  # remove linebreaks
    text = text.strip() # Remove leading whitespace
    return text

df["message"] = df["message"].progress_apply(_unify_text)
df = df[df.message != ""]
df = df[~df.message.isna()]
df = df[~df.message.duplicated()]
df["approx_len"] = df["message"].progress_apply(len)

dataset = Dataset.from_pandas(df)
# Push to HF Hub
repo_name = "TomData/tg-topic-dataset"
dataset.push_to_hub(repo_name, private=True)




