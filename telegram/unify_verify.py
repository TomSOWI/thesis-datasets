import pandas as pd
from tqdm import tqdm
from src.utils import investigate_patterns
from tqdm import tqdm



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


    print("=====After Topic filter=====")
    df = pd.read_parquet(f"{INPUT_PATH}/TG_topic_augmented.parquet")
    # Check processing worked as intended
    investigate_patterns(df,text_column="message",n_examples=5)



if __name__ == "__main__":
    main()