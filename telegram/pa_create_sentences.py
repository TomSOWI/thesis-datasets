import uuid
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import ast
import pyarrow as pa
import pyarrow.parquet as pq

INPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"
OUTPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"
PARQUET_OUTPUT_FILE = f"{OUTPUT_PATH}/TG_sentences.parquet"

# Validate input data
def _validate_dataframe(df):
    with open(f"{INPUT_PATH}/valid_ids.txt", "r") as f:
        ids = set(ast.literal_eval(f.read()))
    tg_ids = set(df["id"].tolist())
    assert len(ids - tg_ids) == 0, f"Missing IDs: {ids - tg_ids}"

# Sentence splitting
def _split_into_sentences(text):
    return sent_tokenize(text)

# Streaming main
def main():
    # print("Loading mapping...")
    #topic_mapping = pd.read_csv(f"{INPUT_PATH}/ch_to_topic_mapping.csv")
    #    #df_sentences = df_sentences.merge(topic_mapping, how="left", left_on="channel_id", right_on="ch_ID")

    # Create ParquetWriter once with schema
    writer = None

    print("Processing messages...")
    df = pd.read_parquet(f"{INPUT_PATH}/TG_target_new.parquet", engine="pyarrow")

    batch = []
    batch_size = 10000

    for _, row in tqdm(df.iterrows(), total=len(df)):

        sentences = _split_into_sentences(str(row["message"]))
        sentence_ids = [str(uuid.uuid4()) for _ in range(len(sentences))]
        for s_id, sent in zip(sentence_ids, sentences):
            batch.append({
                "sentence_id": s_id,
                "sentence": sent,
                "channel_id": row["id"],
                "message_id": row["message_id"]
            })
        
        if len(batch) >= batch_size:
            print(f"Writing {len(batch)} rows to Parquet")
            df_batch = pd.DataFrame(batch)
            table = pa.Table.from_pandas(df_batch)
            if writer is None:
                writer = pq.ParquetWriter(PARQUET_OUTPUT_FILE, table.schema)
            writer.write_table(table)
            batch = []

    if batch:
        print(f"Writing {len(batch)} rows to Parquet")
        batch_df = pd.DataFrame(batch)
        table = pa.Table.from_pandas(batch_df)
        if writer is None:
            writer = pq.ParquetWriter(PARQUET_OUTPUT_FILE, table.schema)
        writer.write_table(table)
        print(f"Writing sentences to parquet finished")


if __name__ == "__main__":
    main()
