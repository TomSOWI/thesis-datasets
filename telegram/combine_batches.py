import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from tqdm import tqdm
import uuid
import os

# Kaggle
import kagglehub
from kagglehub import KaggleDatasetAdapter
import json

INPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"
OUTPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"


def download_from_kaggle():
    # Set this to where your kaggle.json is saved
    kaggle_json_path = "/home/nimtsspi/thesis-datasets/telegram/src/kaggle.json"

    with open(kaggle_json_path) as f:
        creds = json.load(f)

    os.environ["KAGGLE_USERNAME"] = creds["username"]
    os.environ["KAGGLE_KEY"] = creds["key"]

    save_path = "/scratch/usr/nimtsspi/datasets/TG"
    # Set the path to the file you'd like to load
    file_paths = [
        "messages_11862.parquet",
        "messages_15816.parquet",
        "messages_19770.parquet",
        "messages_3954.parquet",
        "messages_7908.parquet"
    ]
    dfs = []

    for file_path in file_paths:
    # Load the latest version
        df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "tomrobinklotz/tg16-07-25",
        file_path,
        # Provide any additional arguments like 
        # sql_query or pandas_kwargs. See the 
        # documenation for more information:
        # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
        )
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    topic_mapping = pd.read_csv(f"{INPUT_PATH}/ch_to_topic_mapping.csv")
    # Enhance with topic information
    df = pd.merge(df, topic_mapping, how="left", left_on="channel_id", right_on="ch_ID")
    # Enhance with message_id 
    df["message_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    df.to_parquet(f"{OUTPUT_PATH}/TG_base.parquet")


def combine_batches():

    file_paths = [
        "/scratch/usr/nimtsspi/datasets/TG/batches/messages_11862.parquet",
        "/scratch/usr/nimtsspi/datasets/TG/batches/messages_15816.parquet",
        "/scratch/usr/nimtsspi/datasets/TG/batches/messages_19770.parquet",
        "/scratch/usr/nimtsspi/datasets/TG/batches/messages_3954.parquet",
        "/scratch/usr/nimtsspi/datasets/TG/batches/messages_7908.parquet"
    ]

    writer = None
    topic_mapping = pd.read_csv(f"{INPUT_PATH}/ch_to_topic_mapping.csv")

    for path in tqdm(file_paths):
        print(f"Reading {path}")
        df = pd.read_parquet(path)

        # Enhance with topic information
        df = pd.merge(df, topic_mapping, how="left", left_on="channel_id", right_on="ch_ID")
        # Enhance with message_id 
        df["message_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

        # Convert to PyArrow for efficient writing
        table = pa.Table.from_pandas(df)
        #if writer is None:
        writer = pq.ParquetWriter(f"{OUTPUT_PATH}/TG_target.parquet", table.schema) #compression="snappy" --default
        writer.write_table(table)
        # free memory
        del df, table  



#if writer:
    #writer.close()

# path = f"{OUTPUT_PATH}/TG_target_new.parquet" #file_paths[0]
# print(f"Reading {path}")
# df = pd.read_parquet(path)
# print(df.iloc[0]["message_id"])
# df["message_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
# df.to_parquet(f"{OUTPUT_PATH}/TG_target_new.parquet")

def main():
    download_from_kaggle()


if __name__ == "__main__":
    main()
