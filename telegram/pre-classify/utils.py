import os
import torch
import pandas as pd
import wandb
import time
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


def wandb_init(
    model: str,
    task: str,
    batch_size: int = 32,
    dataset: str = "TG_unified.parquet",
    max_length: int = 512,
):
    """
    Initialize Weights & Biases run with configurable parameters.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wandb.init(
        project=f"tg-preclassify-{task}-{model.split('/')[-1]}",
        config = {
        "batch_size": batch_size,
        "model": model,
        "dataset": dataset,
        "task": task,
        "max_length": max_length,
        "device": device,
        },
    )
    config = wandb.config
    return config


def load_model(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    model = AutoModelForSequenceClassification.from_pretrained(config.model)
    return model, tokenizer


def single_label_classification(messages, config):
    # Load tokenizer and model
    model, tokenizer = load_model(config)
    model = model.to(config.device)
    model.eval()

    # id2label mapping
    print("Available labels:", model.config.id2label)
    id2label = model.config.id2label

    predictions = []

    for i in range(0, len(messages), config.batch_size):
        batch_texts = messages[i : i + config.batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_length,
        ).to(config.device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1)

        predicted_classes = torch.argmax(probs, dim=1).tolist()
        predictions.extend([id2label[i] for i in predicted_classes])

    print(f"Predictions using {config.model} have finished.")
    return predictions


def main(model, task):

    config = wandb_init(
        model=model,
        task=task
    )

    # Enable faster matmul using TF32 on A100
    set_A100_precission()

    # Load TG dataset
    df = pd.read_parquet(f"{INPUT_PATH}/{config.dataset}.parquet")
    messages = df.message.to_list()

    # Run classification
    start = time.time()
    df["hate"] = single_label_classification(messages, config)
    end = time.time()

    # Check predictions
    print("Values:", df.hate.unique())

    # Save message_id + label
    df = df[["message_id", "hate"]]
    print("Saving final results")
    df.to_parquet(f"{OUTPUT_PATH}/TG_{config.task}.parquet")

    # Log runtime
    print("Runtime (minutes):", (end - start) // 60)
