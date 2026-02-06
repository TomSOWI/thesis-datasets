import os
import torch
import pandas as pd
import wandb
import time
import psutil
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForSequenceClassification

base_path = "/mnt/vast-kisski/projects/kisski_tegami"
INPUT_PATH = f"{base_path}/datasets/TG"
OUTPUT_PATH = f"{base_path}/datasets/TG/labels"


def set_A100_precission():
    # Enable faster matmul using TF32 on A100
    torch.set_float32_matmul_precision("high")
    print("Available GPUs:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.current_device())
    print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))



def load_model(config):
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    model = AutoModelForSequenceClassification.from_pretrained(config["model"])
    # tokenizer = AutoTokenizer.from_pretrained(config.model)
    # model = AutoModelForSequenceClassification.from_pretrained(config.model)
    return model, tokenizer


def single_label_classification(messages, config):
    # Load tokenizer and model
    model, tokenizer = load_model(config)
    # model = model.to(config.device)
    model = model.to(config["device"])
    model.eval()

    # id2label mapping
    #print("Available labels:", model.config.id2label)
    id2label = model.config.id2label

    predictions = []

    # for i in range(0, len(messages), config.batch_size):
    #     batch_texts = messages[i : i + config.batch_size]
    for i in range(0, len(messages), config["batch_size"]):
        batch_texts = messages[i : i + config["batch_size"]]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config["max_length"],
        ).to(config["device"]) ### change!!!!!!!!!!!!!!!!

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1)

        predicted_classes = torch.argmax(probs, dim=1).tolist()
        predictions.extend([id2label[i] for i in predicted_classes])

    print(f"Predictions using {config['model']} have finished.")
    return predictions

def stream_parquet(path, batch_size=50_000):
    """Stream parquet in row-grouped batches using pyarrow."""
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield batch.to_pandas()


def main(model, task):
    print("Memory usage (GB):", psutil.virtual_memory().used / 1e9)

    # config = wandb_init(
    #     model=model,
    #     task=task
    # )

    config = {
        "batch_size": 32,
        "model": model,
        "dataset": "TG_unified",
        "task": task,
        "max_length": 512,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
       }
     
    print("Memory usage (GB):", psutil.virtual_memory().used / 1e9)

    # Enable faster matmul using TF32 on A100
    # set_A100_precission()
    # parquet_path = f"{INPUT_PATH}/{config.dataset}.parquet"
    # output_path = f"{OUTPUT_PATH}/TG_{config.task}.parquet"
    parquet_path = f"{INPUT_PATH}/{config['dataset']}.parquet"
    output_path = f"{OUTPUT_PATH}/TG_{config['task']}.parquet"
    

    start = time.time()

    # Process in streaming chunks
    results = []
    for i, chunk_df in enumerate(stream_parquet(parquet_path, batch_size=50_000)): #50_000
        print(f"Processing chunk {i} with {len(chunk_df)} rows...")

        messages = chunk_df.message.to_list()
        chunk_df[task] = single_label_classification(messages, config)
        results.append(chunk_df[["message_id", task]])

    final_df = pd.concat(results, ignore_index=True)
    final_df.to_parquet(output_path, engine="pyarrow", index=False, compression="snappy")

    end = time.time()
    print("Saved results to:", output_path)
    print("Runtime (minutes):", (end - start) / 60)


