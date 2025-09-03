
#DELE????!!!

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datasets import Dataset, DatasetDict, load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
from tqdm import tqdm

login(token="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")


# Load data
ds = load_dataset("TomData/CMSB")
train_df = ds["train"].to_pandas()
test_df = ds["test"].to_pandas()
eval_df = ds["validation"].to_pandas() 

df = pd.concat([
    train_df.assign(split="train"),
    test_df.assign(split="test"),
    eval_df.assign(split="eval")
], ignore_index=True)
df

len(df.sexist == True)

n_true = len(df[df["sexist"] == True])
df_false = df[df["sexist"] == False].sample(n_true)
df_true = df[df["sexist"] == True]

df_balanced = pd.concat([df_false, df_true], ignore_index=True)
sentences = df_balanced["text"].to_list()

# Parameters
#model_id = "cirimus/modernbert-large-bias-type-classifier"  # Replace with your actual model ID
model_id = "maximuspowers/bias-type-classifier"
batch_size = 16
threshold = 0.5  # you can adjust this based on precision/recall tradeoff
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
model = model.to(device)
model.eval()

# id2label mapping
id2label = model.config.id2label  # e.g., {0: 'spam', 1: 'scam', 2: 'health', ...}

# Store results
predictions = []
predictions_probs = []

for i in tqdm(range(0, len(sentences), batch_size), desc="Multi-label Annotation"):
    batch_texts = sentences[i:i + batch_size]
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits


    probs = torch.sigmoid(logits)  # ← Multi-label uses sigmoid
    predictions_probs.extend(probs)

    
    # Apply threshold
    batch_preds = []
    for row in probs:
        label_idxs = (row > threshold).nonzero(as_tuple=True)[0].tolist()
        labels = [id2label[i] for i in label_idxs]
        batch_preds.append(labels)

    predictions.extend(batch_preds)


df_balanced["label"] = predictions
for label in model.config.id2label.values():
    if label == "gender":
        df_balanced[label] = [label in row["label"] for _, row in tqdm(df_balanced.iterrows(), total=len(df_balanced))]

df_balanced


# Eval
(df_balanced["gender"] == df_balanced["sexist"]).mean()



