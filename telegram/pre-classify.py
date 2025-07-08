import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
from datasets import load_dataset, DatasetDict, Dataset, load_from_disk
import os
import torch
import pandas as pd
from tqdm import tqdm
import wandb
import time


os.environ["WANDB_MODE"] = "offline"
wandb.init(project="tg-pre-classify")

# Enable faster matmul using TF32 on A100
torch.set_float32_matmul_precision('high')
print("Available GPUs:", torch.cuda.device_count())
print("Current GPU:", torch.cuda.current_device())
GPU = torch.cuda.get_device_name(torch.cuda.current_device())
print("GPU Name:", GPU)


tqdm.pandas()
BATCH_SIZE = 32
GENDER_MODEL = "bias-type-classifier"
SENTI_MODEL = "twitter-roberta-base-sentiment-latest"
HATE_MODEL = "bert-base-uncased-hatexplain"
POLITIC_MODEL = "magpie-babe-ft-xlm"
LEXBIAS_MODEL = ""


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
    #predictions_probs = []

    #for i in tqdm(range(0, len(messages), BATCH_SIZE), desc="Multi-label Annotation"):
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


def single_label_classification(messages, max_length, model_id, device):
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
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1)

        #prediction_probs.extend(probs.cpu().tolist())
        predicted_classes = torch.argmax(probs, dim=1).tolist()
        predictions.extend([id2label[i] for i in predicted_classes])

    print(f"Predictions using {model_id} has finished")
    return predictions


def main():
    #ds = load_from_disk("/scratch/usr/nimtsspi/datasets/tg-focused-topic-dataset-280len-limit") # already annotated for gender
    #df = ds["train"].to_pandas()
    #df = pd.read_parquet("/scratch/usr/nimtsspi/datasets/tg-focused-topic-dataset-280len-limit/df.parquet")
    #print(df.columns)

    #df["gender"] = annotate_gender(df)
    #df["hate"] = annotate_hate(df)
    #df.to_parquet("/scratch/usr/nimtsspi/datasets/tg-focused-topic-dataset-280len-limit/df.parquet")
    #df["sentiment"] = annotate_senti(df)
    #df.to_parquet("/scratch/usr/nimtsspi/datasets/tg-focused-topic-dataset-280len-limit/df.parquet")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Classify TG
    tg_data_path = "/scratch/usr/nimtsspi/datasets/tg-focused-topic-dataset-280len-limit"
    tg_ds = load_from_disk(tg_data_path) 
    tg_df = tg_ds["train"].to_pandas()
    tg_messages = tg_df["message"].to_list()

    # Classify
    start = time.time()
    tg_df["gender"] = multi_label_classification(tg_messages, model_id=GENDER_MODEL, target_label="gender",device=device)
    tg_df["hate"] = single_label_classification(tg_messages, model_id=HATE_MODEL, device=device)
    tg_df["senti"] = single_label_classification(tg_messages, model_id=SENTI_MODEL,device=device)
    tg_df["lexbias"] = single_label_classification(tg_messages, model_id=LEXBIAS_MODEL,device=device)
    end = time.time()

    # Save
    tg_df.to_parquet("/scratch/usr/nimtsspi/datasets/TG-preclassify/tg-280lim-preclassify.parquet") #!!!

    # Loag Runtime and GPU
    wandb.log({
        "gpu": GPU,
        "runtime_seconds": (end - start)//60
    })



if __name__ == "__main__":
    main()