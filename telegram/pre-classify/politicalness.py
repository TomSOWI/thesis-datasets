import os
import torch
import pandas as pd
import wandb
import time
from transformers import pipeline

os.environ["WANDB_MODE"] = "offline"
wandb.init(
    project="tg-pre-classify-politicalness",
    config={
        "batch_size": 32,
        "model": "Political_DEBATE_large_v1.0",
        "task": "single-label-classification",
        "dataset": "TG_unified.parquet",
        "max_length": 512,
    },
)

# Access the config
config = wandb.config
# Define global parameters
INPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"
OUTPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG/labels"
BATCH_SIZE = config.batch_size
MODEL = config.model
MAX_LENGTH = config.max_length

# Enable faster matmul using TF32 on A100
torch.set_float32_matmul_precision("high")
print("Current GPU:", torch.cuda.current_device())
print("Available GPUs:", torch.cuda.device_count())
print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))


def zero_shot_classification(messages, model_id, device):
    model_path = f"/scratch/usr/nimtsspi/models/{model_id}"
    pipe = pipeline("zero-shot-classification", model=model_path, tokenizer=model_path, batch_size = BATCH_SIZE, device=device)
    hypothesis_template = 'This text is {} about politics.'
    labels = ["is not", "is"]
    mapping = {"is not": "NOT POLITICAL", "is": "POLITICAL"} # like in the paper
    results = pipe(messages, labels, hypothesis_template = hypothesis_template, multi_label = True)
    labels = [mapping[label['labels'][0]] for label in results]
    return labels

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load TG
    df = pd.read_parquet(f"{INPUT_PATH}/TG_unified.parquet")
    messages = df.message.to_list()

    # Classify
    start = time.time()
    df["politicalness"] = zero_shot_classification(messages, model_id=MODEL, device=device)
    end = time.time()

    # Check
    print("Values:", df.politicalness.value_counts())
   
    # Save message_id + label
    df = df[["message_id", "politicalness"]]
    print("Saving final results")
    df.to_parquet(f"{OUTPUT_PATH}/TG_politicalness.parquet")

    # Log Runtime
    print("Runtime:", (end - start) // 60)

if __name__ == "__main__":
    main()