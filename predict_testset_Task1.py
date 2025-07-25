import pandas as pd
import numpy as np
import torch
import re
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, XLMRobertaForSequenceClassification
from sklearn.metrics import f1_score
from tqdm import tqdm

def get_latest_checkpoint(path):
    checkpoints = [d for d in os.listdir(path) if re.match(r"checkpoint-\d+", d)]
    if not checkpoints:
        return path
    latest = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    return os.path.join(path, latest)

def clean_text(text):
    text = re.sub(r'@User|<user>', '', text)
    text = re.sub(r'http\S+|www\.\S+', '[URL]', text)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class TestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding='max_length',
                                  max_length=self.max_length, return_tensors="pt")
        return {key: val.squeeze() for key, val in encoding.items()}

best_thresholds = {
    "en": 0.70,
    "de": 0.70,
    "fr": 0.58,
    "ru": 0.70,
}

test_path = "/users/PAS2836/krishnakb/ondemand/krishna_proj/Task1/test_data_SMM4H_2025_Task_1_no_labels.csv"
test_df = pd.read_csv(test_path)
test_df["text"] = test_df["text"].apply(clean_text)

model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

all_test_preds = []

for lang in test_df["language"].unique():
    print(f"\n=== Predicting for Language: {lang} ===")

    lang_df = test_df[test_df["language"] == lang].reset_index(drop=True)
    if lang_df.empty:
        continue

    dataset = TestDataset(lang_df["text"].tolist(), tokenizer)

    model_path = get_latest_checkpoint(f"./results_{lang}")
    if not os.path.exists(model_path):
        print(f"Model path not found for {lang}, skipping...")
        continue

    model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
    trainer = Trainer(model=model, tokenizer=tokenizer)

    predictions = trainer.predict(dataset)
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()

    threshold = best_thresholds.get(lang, 0.5)
    lang_df["prediction"] = (probs >= threshold).astype(int)

    lang_df["predicted_label"] = lang_df["prediction"]
    all_test_preds.append(lang_df[["id", "predicted_label"]])

# Save final submission
if all_test_preds:
    final_df = pd.concat(all_test_preds, ignore_index=True)
    final_df = final_df.sort_values("id").reset_index(drop=True)
    final_df.to_csv("final_submission.csv", index=False)
    print("\n✅ Submission file saved as final_submission.csv")
else:
    print("⚠️ No predictions were generated — check model paths and test data.")