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
    project="tg-pre-classify-gender",
    config={
        "batch_size": 32,
        "model": "bias-type-classifier",
        "task": "multi-label-classification",
        "dataset": "TG_unified.parquet",
        "max_length": 512
    }
)

# Access the config
config = wandb.config
BATCH_SIZE = config.batch_size
MODEL = config.model
MAX_LENGTH = config.max_length

# Enable faster matmul using TF32 on A100
torch.set_float32_matmul_precision('high')
print("Current GPU:", torch.cuda.current_device())
print("Available GPUs:", torch.cuda.device_count())
print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))


def load_model(model_id):
    model_path = f"/scratch/usr/nimtsspi/models/{model_id}"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True) 

    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    return model, tokenizer

def multi_label_classification(messages, model_id, target_label, device):
    # Load tokenizer and model
    model, tokenizer = load_model(model_id)
    model = model.to(device)
    model.eval()

    # id2label mapping
    print(model.config.id2label.values())
    id2label = model.config.id2label 

    # Set threshold
    threshold = 0.5  # you can adjust this based on precision/recall tradeoff
  
    # Store results
    predictions = []
 
    for i in range(0, len(messages), BATCH_SIZE):
        batch_texts = messages[i:i + BATCH_SIZE]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits


        probs = torch.sigmoid(logits)  # ← Multi-label uses sigmoid
        #predictions_probs.extend(probs)

        # Apply threshold
        batch_preds = []
        for row in probs:
            label_idxs = (row > threshold).nonzero(as_tuple=True)[0].tolist()
            labels = [id2label[i] for i in label_idxs]
            batch_preds.append(labels)

        predictions.extend(batch_preds)

        # Convert Multi Label to single Label
    print(f"Predictions using {model_id} has finished")
    single_label_predictions = [target_label in prediction for prediction in predictions] 
    return single_label_predictions


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load TG
    df = pd.read_parquet(f"{INPUT_PATH}/TG_unified.parquet")
    messages = df.message.to_list()
    
    # Classify
    start = time.time()
    df["gender"] = multi_label_classification(messages, model_id=MODEL, target_label="gender",device=device)
    end = time.time()

    # Check
    print("Values:", df.gender.unique())

    # Save message_id + label
    df = df[["message_id", "gender"]]
    print("Saving final results")
    df.to_parquet(f"{OUTPUT_PATH}/TG_gender.parquet") #!!!

    # Log Runtime 
    print("Runtime:", (end - start)//60)


if __name__ == "__main__":
    main()