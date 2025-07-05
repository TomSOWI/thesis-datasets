import uuid
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import sent_tokenize


import ast
from src.utils import to_parquet

INPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"
OUTPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"
tqdm.pandas()

def _validate_dataframe(df):
    # Validate TG dataset by checking that all relevant channel ids are pressent in the dataframe
    with open("/scratch/usr/nimtsspi/datasets/TG/valid_ids.txt", "r") as f:
        ids = f.read()

    ids = set(ast.literal_eval(ids))
    tg_ids = set(df.id.to_list())
    # Validation 
    assert len(ids - tg_ids) == 0

def _split_into_sentences(text):
    """
    Splits an text into sentences and filters for english sentences.

    Args:
        text (str): The text to be split.

    Returns:
        list: The list of sentences.
    """
    sentences = sent_tokenize(text)

    # def en_sentence(sentence):
    #     try:
    #         lang = detect(sentence)
    #     except:
    #         print(f"Error for: {sentence}")

    #     if lang == "en":
    #         return True
    #     else:
    #         return False
            

    # sentences = [
    #     sentence for sentence in sentences if en_sentence(sentence)
    # ]
    return sentences



@to_parquet(f"{OUTPUT_PATH}/TG_sentences.parquet")
def main():
    channel_df = pd.read_parquet(f"{INPUT_PATH}/TG_target.parquet")
    # Ensure dataframe contains all relevant ids
    _validate_dataframe(channel_df)

    # split into sentences
    sentence_dataframes = []
    for _, channel_row in tqdm(channel_df.iterrows()):
        #sentences = _split_into_sentences(channel_row["message"])
        sentences = sent_tokenize(channel_row["message"])

        sentence_ids = [str(uuid.uuid4()) for _ in range(len(sentences))]
        sentence_dataframes.append(
            pd.DataFrame(
                {
                    "sentence": sentences,
                    "channel_id": [channel_row["id"]] * len(sentences),
                    "sentence_id": sentence_ids,
                },
            ),
        )

    df = pd.concat(sentence_dataframes)
    channel_topic_df = pd.read_csv(f"{INPUT_PATH}/ch_to_topic_mapping.csv")
    df = pd.merge(df, channel_topic_df, how="left",left_on="channel_id", right_on="ch_ID")

    return df

if __name__ == "__main__":
    main()




