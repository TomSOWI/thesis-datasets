import os
import torch
import pandas as pd
import wandb
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import psutil


# INPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"
# OUTPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG/labels"

BASE_PATH = "/mnt/vast-kisski/projects/kisski_tegami"
INPUT_PATH = f"{BASE_PATH}/datasets/TG"
OUTPUT_PATH = f"{BASE_PATH}/datasets/TG/labels"

os.environ["WANDB_MODE"] = "offline"
wandb.init(
    project="tg-pre-classify",
    config={
        "batch_size": 32,
        "model": os.environ["MODEL"],
        "task": os.environ["TASK"],
        "dataset": "TG_unified.parquet",
        "max_length": 512,
    },
)

# Access the config
config = wandb.config
BATCH_SIZE = config.batch_size
MODEL = config.model #"mediabiasgroup/magpie-babe-ft"
MAX_LENGTH = config.max_length
TASK = config.task

# Enable faster matmul using TF32 on A100
# torch.set_float32_matmul_precision("high")
# print("Current GPU:", torch.cuda.current_device())
# print("Available GPUs:", torch.cuda.device_count())
# print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))


def load_model(model_id):
    model_path = f"{BASE_PATH}/models/{model_id}"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    )
    return model, tokenizer


def single_label_classification(messages, model_id, device):
    # Load tokenizer and model
    model, tokenizer = load_model(model_id)
    model = model.to(device)
    model.eval()

    # id2label mapping
    print(model.config.id2label.values())
    id2label = model.config.id2label

    predictions = []

    for i in range(0, len(messages), BATCH_SIZE):
        batch_texts = messages[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1)

        # prediction_probs.extend(probs.cpu().tolist())
        predicted_classes = torch.argmax(probs, dim=1).tolist()
        predictions.extend([id2label[i] for i in predicted_classes])

    print(f"Predictions using {model_id} has finished")
    return predictions


def main():
    print("Memory usage (GB):", psutil.virtual_memory().used / 1e9)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load TG
    df = pd.read_parquet(f"{INPUT_PATH}/TG_unified.parquet")
    messages = df.message.to_list()
    print("Memory usage (GB):", psutil.virtual_memory().used / 1e9)
    # Classify
    start = time.time()
    df[TASK] = single_label_classification(messages, model_id=MODEL, device=device)
    end = time.time()

    # Check
    print("Values:", df[TASK].unique())

    # Save message_id + label
    df = df[["message_id", TASK]]
    print("Saving final results")
    df.to_parquet(f"{OUTPUT_PATH}/TG_{TASK}.parquet")
    print("Memory usage (GB):", psutil.virtual_memory().used / 1e9)
    # Log Runtime
    print("Runtime:", (end - start) // 60)


if __name__ == "__main__":
    main()
