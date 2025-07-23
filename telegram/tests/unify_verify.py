import pandas as pd
from tqdm import tqdm
import re
from src.utils import to_parquet
from markdown_text_clean import clean_text
from tqdm import tqdm
tqdm.pandas()

INPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"
OUTPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"
tqdm.pandas()

def remove_links(text):
    text = re.sub(r'https?://\S+', '', text)
    #text = re.sub(r"https?://[A-Za-z0-9./]+", "", text) 
    return text

def count_no_english_chars(text):
    # Define characters NOT in ASCII or emoji ranges
    pattern = re.compile(
        r'[^\x00-\x7F'                             # ASCII
                                                  # Apostrophe '
        r'\U0001F300-\U0001F5FF'                   # Misc Symbols and Pictographs
        r'\U0001F600-\U0001F64F'                   # Emoticons
        r'\U0001F680-\U0001F6FF'                   # Transport and Map
        r'\U0001F700-\U0001F77F'                   # Alchemical Symbols
        r'\U0001F780-\U0001F7FF'                   # Geometric Shapes Extended
        r'\U0001F800-\U0001F8FF'                   # Supplemental Arrows-C
        r'\U0001F900-\U0001F9FF'                   # Supplemental Symbols and Pictographs
        r'\U0001FA00-\U0001FA6F'                   # Chess Symbols, etc.
        r'\U0001FA70-\U0001FAFF'                   # Symbols and Pictographs Extended-A
        r'\U00002700-\U000027BF'                   # Dingbats
        r']+',
        flags=re.UNICODE
    )
    asian_chars_pattern = re.compile(
    r'['
    r'\u3040-\u309F'   # Hiragana
    r'\u30A0-\u30FF'   # Katakana
    r'\u4E00-\u9FFF'   # CJK Unified Ideographs
    r'\uAC00-\uD7AF'   # Hangul Syllables
    r'\u1100-\u11FF'   # Hangul Jamo
    r'\u3130-\u318F'   # Hangul Compatibility Jamo
    r']+',
    flags=re.UNICODE
)

    # Find all matching substrings (groups of chars), then flatten into characters
    matches = asian_chars_pattern.findall(text)
    characters = ''.join(matches)
    return len(characters)


def main():
    df = pd.read_parquet(f"{INPUT_PATH}/TG_unified.parquet")
    df["ASCII"] = df["message"].progress_apply(count_no_english_chars)

    # # Gather texts with links
    # test1_df = df[df.message.str.contains("http")]
    # ids = test1_df["message_id"].to_list()

    # # Processing only for texts with links
    # df["message"]  = df["message"].apply(remove_links)
    # df = df[df["message"].notna() & (df["message"] != "")]
    # df = df.drop_duplicates(subset=["message"])
    # print(f"Length remove-links http: {len(df)}")
    # df = df[df.message_id.isin(ids)]
    # print(f"Length non remove-links http: {len(df)}")

    # # Comparision
    # # Without Links
    # df =  df.sample(10)
    # no_link_message = df.message.to_list()
    # no_link_ids = df.message_id.to_list()

    # # With Links
    # link_message = test1_df[test1_df.message_id.isin(no_link_ids)].message.to_list()
    # print("--------")
    # [print(f"{idx}.:{m}") for idx, m in enumerate(link_message)]
    # # Without Links
    # [print(f"{idx}.:{m}") for idx, m in enumerate(no_link_message)]
  

    START = 0
    END = START + 100

    www_message = df[df.ASCII > 0].message.to_list()
    print(len(www_message))
    [print(f"{idx}.:{m}") for idx, m in enumerate(www_message[START:END])]


    # www_message = df[df.message.str.contains("https")].message.to_list()
    # print(len(www_message))
    # [print(f"{idx}.:{m}") for idx, m in enumerate(www_message[START:END])]
    # print("---------------")
    # www_message = df[df.message.str.contains(r"www.", regex=True, na=False)].message.to_list()
    # print(len(www_message))
    # [print(f"{idx}.:{m}") for idx, m in enumerate(www_message[START:END])]
    # print("---------------")
    # www_message = df[df.message.str.contains("bit.ly")].message.to_list()
    # print(len(www_message))
    # [print(f"{idx}.:{m}") for idx, m in enumerate(www_message[START:END])]
    # print("---------------")
    # www_message = df[df.message.str.contains("@ ")].message.to_list()
    # print(len(www_message))
    # [print(f"{idx}.:{m}") for idx, m in enumerate(www_message[START:END])]
    # print("---------------")
    # www_message = df[df.message.str.contains("\*")].message.to_list()
    # print(len(www_message))
    # [print(f"{idx}.:{m}") for idx, m in enumerate(www_message[START:END])]
    # print("---------------")
    # www_message = df[df.message.str.contains("bit.ly")].message.to_list()
    # print(len(www_message))
    # [print(f"{idx}.:{m}") for idx, m in enumerate(www_message[START:END])]
    # print("---------------")
    # www_message = df[df.message.str.contains(r'<.*?>', regex=True, na=False)].message.to_list()
    # print(len(www_message))
    # [print(f"{idx}.:{m}") for idx, m in enumerate(www_message[START:END])]
    # print("---------------")
    # www_message = df[df.message.str.contains("&amp;")].message.to_list()
    # print(len(www_message))
    # [print(f"{idx}.:{m}") for idx, m in enumerate(www_message[START:END])]
    # print("---------------")
    # www_message = df[df.message.str.contains("&lt;")].message.to_list()
    # print(len(www_message)) #4187
    # [print(f"{idx}.:{m}") for idx, m in enumerate(www_message[START:END])]
    # print("---------------")
    # www_message = df[df.message.str.contains("&#123")].message.to_list()
    # print(len(www_message)) #4187
    # [print(f"{idx}.:{m}") for idx, m in enumerate(www_message[START:END])]
    # print("---------------")
    # test3_df = df[df.message.str.contains("__")]
    # test4_df = df[df.message.str.contains("\*\*")]



if __name__ == "__main__":
    main()