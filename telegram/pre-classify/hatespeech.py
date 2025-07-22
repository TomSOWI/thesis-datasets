import os
import torch
import pandas as pd
import wandb
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

INPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"
OUTPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG/labels"

#BATCH_SIZE = 32
#GENDER_MODEL = "bias-type-classifier"
#SENTI_MODEL = "twitter-roberta-base-sentiment-latest"
#HATE_MODEL = "bert-base-uncased-hatexplain"
#POLITIC_MODEL = "magpie-babe-ft-xlm"
#LEXBIAS_MODEL = ""

os.environ["WANDB_MODE"] = "offline"
wandb.init(
    project="tg-pre-classify-hate",
    config={
        "batch_size": 32,
        "model": "bert-base-uncased-hatexplain",
        "task": "single-label-classification",
        "dataset": "TG_unified.parquet",
        "max_length": 512
    }
)

# Access the config
config = wandb.config
BATCH_SIZE = config.batch_size
SENTI_MODEL = config.model
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

def single_label_classification(messages, model_id, device):
    # Load tokenizer and model
    model, tokenizer = load_model(model_id)
    model = model.to(device)
    model.eval()
 
    # id2label mapping
    print(model.config.id2label.values())
    id2label = model.config.id2label 

    predictions = []
    #prediction_probs = []

    #for i in tqdm(range(0, len(messages), BATCH_SIZE), desc="Hate Speech Classification"):
    for i in range(0, len(messages), BATCH_SIZE):
        batch_texts = messages[i:i + BATCH_SIZE]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1)

        #prediction_probs.extend(probs.cpu().tolist())
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
    df["hate"] = single_label_classification(messages, model_id=SENTI_MODEL,device=device)
    end = time.time()

    # Check
    print("Values:", df.hate.unique())

    # Save message_id + label
    df = df[["message_id", "hate"]]
    print("Saving final results")
    df.to_parquet(f"{OUTPUT_PATH}/TG_hate.parquet") #!!!

    # Log Runtime 
    print("Runtime:", (end - start)//60)


if __name__ == "__main__":
    main()