import pandas as pd
from tqdm import tqdm
import re
from src.utils import to_parquet
from langdetect import detect, LangDetectException

INPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"
OUTPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"
tqdm.pandas()

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

# def safe_detect(x):
#     try:
#         #if isinstance(x, str) and x.strip():  # not empty and is string
#         return detect(x) == "en"
#     except LangDetectException:
#         return "LangDetectError"
#     #return False  # fallback: treat as non-English if undetectable

def safe_detect(x):
    try:
        return "en" if detect(x) == "en" else "non-en"
    except LangDetectException:
        return "LangDetectError"

@to_parquet(f"{OUTPUT_PATH}/TG_filtered_sentences.parquet")
def main():
    df = pd.read_parquet(f"{INPUT_PATH}/TG_sentences.parquet")
    df["sentence"] = df["sentence"].progress_apply(_unify_text)
    df = df[df.sentence != ""]
    df = df[~df.sentence.isna()]
    df = df[~df.sentence.duplicated()]
    #lang_mask = df["sentence"].progress_apply(lambda x: detect(x) == "en")
    #df = df[lang_mask]
    df["lang_mask"] = df["sentence"].progress_apply(safe_detect)

    # Add meta information

    return df

if __name__ == "__main__":
    main()