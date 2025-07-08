import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from tqdm import tqdm
import uuid

INPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"
OUTPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"

file_paths = [
    "/scratch/usr/nimtsspi/datasets/TG/batches/messages_11862.parquet",
    "/scratch/usr/nimtsspi/datasets/TG/batches/messages_15816.parquet",
    "/scratch/usr/nimtsspi/datasets/TG/batches/messages_19770.parquet",
    "/scratch/usr/nimtsspi/datasets/TG/batches/messages_3954.parquet",
    "/scratch/usr/nimtsspi/datasets/TG/batches/messages_7908.parquet"
]

writer = None
#topic_mapping = pd.read_csv(f"{INPUT_PATH}/ch_to_topic_mapping.csv")

for path in tqdm(file_paths):
    print(f"Reading {path}")
    df = pd.read_parquet(path)
    #print(df.columns)
    #print(topic_mapping.columns)
    #df = pd.merge(df, topic_mapping, how="left", left_on="id", right_on="ch_ID")
    #print(df.columns)
    #break
    # Enhance batch with message_id for sentence-message mapping later
    df["message_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
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