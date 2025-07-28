import os
import torch
import pandas as pd
import wandb
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

INPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"
OUTPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG/labels"

os.environ["WANDB_MODE"] = "offline"
wandb.init(
    project="tg-pre-classify-political_leaning",
    config={
        "batch_size": 32,
        "model": "political-leaning-politics",
        "task": "single-label-classification",
        "dataset": "TG_unified.parquet",
        "max_length": 512,
    },
)

# Access the config
config = wandb.config
BATCH_SIZE = config.batch_size
MODEL = config.model
MAX_LENGTH = config.max_length

# Enable faster matmul using TF32 on A100
torch.set_float32_matmul_precision("high")
print("Current GPU:", torch.cuda.current_device())
print("Available GPUs:", torch.cuda.device_count())
print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))


def load_model(model_id):
    model_path = f"/scratch/usr/nimtsspi/models/{model_id}"
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load TG
    df = pd.read_parquet(f"{INPUT_PATH}/TG_unified.parquet")
    messages = df.message.to_list()


    # Classify
    start = time.time()
    raw_labels = single_label_classification(messages, model_id=MODEL, device=device)
    mapping = {"LABEL_0": "left", "LABEL_1": "center","LABEL_2": "right"}
    df["political_leaning"] = [mapping[label] for label in raw_labels]
    end = time.time()

    # Check
    print("Values:", df.political_leaning.value_counts())

    # Save message_id + label
    df = df[["message_id", "political_leaning"]]
    print("Saving final results")
    df.to_parquet(f"{OUTPUT_PATH}/TG_political_leaning.parquet")

    # Log Runtime
    print("Runtime:", (end - start) // 60)


if __name__ == "__main__":
    main()
