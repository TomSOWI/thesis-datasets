import html.entities
import pandas as pd
import re
from tqdm import tqdm
from src.utils import to_parquet
from markdown_text_clean import clean_text
from langdetect import detect, LangDetectException
import html
tqdm.pandas()

html.escape

INPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"
OUTPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"


#https//t.me/nicekiddbot
#https//:t.me/FreedomAmericanConvoy 
#https : // t . me/sineinjuria/ 51731
#https.//t.me/BINARYKILLEER


# Precompiled regex patterns 
RE_PATTERNS = {
    "rt": re.compile(r"RT\ "),  # Remove "RT" retweet indicator
    "mention": re.compile(r"@[A-Za-z0-9_]+"),  # Replace user mentions with "@user"
    "attachment": re.compile(r'\[.*?\]\(https?://[^\)]+\)'),  # Remove markdown link attachments [text](http://...)
    "url1": re.compile(r'https?:?.?\n?//\n?\S+'),  # Remove any remaining http links incluiding some wrong formatted ones
    "url2": re.compile(r'https?:?.?\n?\s?//\s?\S+/\S+'), # Further wrong formatted ones
    "domain": re.compile(r'www\.[^\s.]+\.[^\s]+'), # Remove domains --> ensure matched string contain at least two "." with non-whitespace characters in between
    "bit_url": re.compile(r'bit.ly/\S+'), # remove bit.ly links
    #"url2": re.compile(r'https?:?.?//[A-Za-z0-9./]+'), #remove remaining links
    "tab": re.compile(r"\t"),  # Replace tabs with space
    "newline": re.compile(r"\n"),  # Replace newline with space
    "carriage": re.compile(r"\r"),  # Replace carriage return with space
    "multi_space": re.compile(r" +"),  # Replace multiple spaces with a single space
    "linebreak": re.compile(r"linebreak"),  # Remove literal word "linebreak"
}

    
    # text = re.sub(r'\[.*?\]\(https?://[^\)]+\)', '', text) # remove attachement links
    # text = re.sub(r'https?://\S+', '', text) # remove remaining links
    # text = re.sub(r"https?://[A-Za-z0-9./]+", "", text)  # remove links


def _unify_text(text):
    # Catch NAs
    if pd.isna(text):
        return text
    text = clean_text(text)
    text = RE_PATTERNS["rt"].sub(" ", text)
    text = RE_PATTERNS["mention"].sub("@user", text)
    text = RE_PATTERNS["attachment"].sub('', text)
    text = RE_PATTERNS["url1"].sub('', text)
    text = RE_PATTERNS["url2"].sub('', text)
    text = RE_PATTERNS["bit_url"].sub('',text)
    text = RE_PATTERNS["domain"].sub('',text)
    text = RE_PATTERNS["tab"].sub(" ", text)
    text = RE_PATTERNS["newline"].sub(" ", text)
    text = RE_PATTERNS["carriage"].sub(" ", text)
    text = RE_PATTERNS["multi_space"].sub(" ", text)
    text = RE_PATTERNS["linebreak"].sub("", text)
    # Resolve broken HTML
    text = html.unescape(text)

    return text.strip()

def safe_detect(x):
    try:
        return detect(x)
    except LangDetectException:
        return "LangDetectError"

@to_parquet(f"{OUTPUT_PATH}/TG_unified.parquet")
def main():
    df = pd.read_parquet(f"{INPUT_PATH}/TG_280limit.parquet")

    # Apply text cleaning function to entire column
    df["message"] = df["message"].progress_apply(_unify_text)

    # Drop empty and NaN messages
    df = df[df["message"].notna() & (df["message"] != "")]
    print(f"N without empty/na: {len(df)}")


    # Drop duplicate messages
    df = df.drop_duplicates(subset=["message"])
    print(f"N without duplicates: {len(df)}")
    
    # Lang Detect: Dropped tooo inaccurate
    #df["language"] = df["message"].progress_apply(safe_detect)

    # Remove sentences only containing non-english characters e.g. emojis, chineese characters
    non_english = re.compile(r'^[^\x00-\x7F]+$')
    df = df[~df.message.str.contains(non_english, regex=True)]
    print(f"N without only foreign signed/emoji messages: {len(df)}")

    print(f"N unified messages: {len(df)}")
    return df

if __name__ == "__main__":
    main()

  