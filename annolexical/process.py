import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os
import torch
import pandas as pd
from tqdm import tqdm
import wandb
import time

tqdm.pandas()


INPUT_PATH = "/scratch/usr/nimtsspi/datasets/annolex"
OUTPUT_PATH = "/scratch/usr/nimtsspi/datasets/annolex"

os.environ["WANDB_MODE"] = "offline"
wandb.init(
    project="annolex-pre-classify",
    config={"batch_size": 32, "dataset": "base.parquet", "max_length": 512},
)

# Access the config
config = wandb.config
BATCH_SIZE = config.batch_size
MAX_LENGTH = config.max_length
DATASET = config.dataset
GENDER_MODEL = "bias-type-classifier"
SENTI_MODEL = "twitter-roberta-base-sentiment-latest"
HATE_MODEL = "bert-base-uncased-hatexplain"
LEXBIAS_MODEL = "magpie-babe-ft-xlm"
POLITICALNESS_MODEL = "Political_DEBATE_large_v1.0"
POLITICAL_LEANING_MODEL = "political-leaning-politics"

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

def zero_shot_classification(messages, model_id, device):
    model_path = f"/scratch/usr/nimtsspi/models/{model_id}"
    pipe = pipeline("zero-shot-classification", model=model_path, tokenizer=model_path, batch_size = BATCH_SIZE, device=device)
    hypothesis_template = 'This text is {} about politics.'
    labels = ["is not", "is"]
    mapping = {"is not": "NOT POLITICAL", "is": "POLITICAL"}
    results = pipe(messages, labels, hypothesis_template = hypothesis_template, multi_label = True)
    print(f"Predictions using {model_id} has finished")
    labels = [mapping[label['labels'][0]] for label in results]
    return labels

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
        batch_texts = messages[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.sigmoid(logits)  # ← Multi-label uses sigmoid
        # predictions_probs.extend(probs)

        # Apply threshold
        batch_preds = []
        for row in probs:
            label_idxs = (row > threshold).nonzero(as_tuple=True)[0].tolist()
            labels = [id2label[i] for i in label_idxs]
            batch_preds.append(labels)

        predictions.extend(batch_preds)

        # Convert Multi Label to single Label
    print(f"Predictions using {model_id} has finished")
    single_label_predictions = [
        target_label in prediction for prediction in predictions
    ]
    return single_label_predictions


def single_label_classification(messages, model_id, device):
    # Load tokenizer and model
    model, tokenizer = load_model(model_id)
    model = model.to(device)
    model.eval()

    # id2label mapping
    print(model.config.id2label.values())
    id2label = model.config.id2label

    predictions = []
    # prediction_probs = []

    # for i in tqdm(range(0, len(messages), BATCH_SIZE), desc="Hate Speech Classification"):
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

    df = pd.read_parquet(f"{INPUT_PATH}/{DATASET}")
    print(len(df))
    sentences = df["text"].to_list()

    # Classify
    start = time.time()

    df["gender"] = multi_label_classification(
        sentences, model_id=GENDER_MODEL, target_label="gender", device=device
    )
    df["hate"] = single_label_classification(
        sentences, model_id=HATE_MODEL, device=device
    )
    df["senti"] = single_label_classification(
        sentences, model_id=SENTI_MODEL, device=device
    )
    df["lexbias"] = single_label_classification(
        sentences, model_id=LEXBIAS_MODEL, device=device
    )
    df["politicalness"] = zero_shot_classification(
        sentences, model_id=POLITICALNESS_MODEL, device=device
    )
    # Political leaning with mapping
    raw_labels = single_label_classification(
        sentences, model_id=POLITICAL_LEANING_MODEL, device=device 
    )
    mapping = {"LABEL_0": "left", "LABEL_1": "center","LABEL_2": "right"}
    df["political_leaning"] = [mapping[label] for label in raw_labels]

    end = time.time()

    # Log Runtime
    print("Runtime:", (end - start) // 60)

    # Save
    df.to_parquet(f"{OUTPUT_PATH}/preclassify.parquet")


if __name__ == "__main__":
    main()
