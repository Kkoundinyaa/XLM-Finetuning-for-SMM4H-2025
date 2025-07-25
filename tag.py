import torch 
import transformers
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer, 
                          TrainingArguments, AutoModel, EarlyStoppingCallback)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from torch.nn import CrossEntropyLoss
from collections import Counter
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import re
import os
from sklearn.model_selection import train_test_split

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

def augment_text(text, tokenizer, mask_prob=0.15):
    tokens = tokenizer.tokenize(text)
    num_masks = max(1, int(len(tokens) * mask_prob))
    masked_indices = random.sample(range(len(tokens)), num_masks)
    for idx in masked_indices:
        tokens[idx] = "[MASK]"
    return tokenizer.convert_tokens_to_string(tokens)

class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', 
                                  max_length=self.max_length, return_tensors="pt")
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    preds = np.argmax(eval_pred.predictions, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average='binary', zero_division=1),
        "recall": recall_score(labels, preds, average='binary', zero_division=1),
        "f1_macro": f1_score(labels, preds, average='macro', zero_division=1),
        "f1": f1_score(labels, preds, average='binary', zero_division=1),
    }

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        unique_labels = torch.unique(labels, sorted=True)
        weight_tensor = torch.ones(2, dtype=torch.float).to(logits.device)

        for label in unique_labels:
            count = (labels == label).sum().item()
            if count > 0:
                weight_tensor[label] = labels.size(0) / (2.0 * count)

        loss_fct = CrossEntropyLoss(weight=weight_tensor)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

train_df = pd.read_csv("/users/PAS2836/krishnakb/ondemand/krishna_proj/Task1/train_data_SMM4H_2025_Task_1.csv")
dev_df = pd.read_csv("/users/PAS2836/krishnakb/ondemand/krishna_proj/Task1/dev_data_SMM4H_2025_Task_1.csv")

languages = train_df['language'].unique()
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

all_dev_predictions = []

for lang in languages:
    print(f"\n==== Training for Language: {lang} ====")
    train_lang = train_df[train_df['language'] == lang].reset_index(drop=True)
    dev_lang = dev_df[dev_df['language'] == lang].reset_index(drop=True)

    if len(train_lang['label'].unique()) < 2:
        print(f"Skipping language '{lang}' — only one class present.")
        continue

    train_lang["text"] = train_lang["text"].apply(clean_text)
    dev_lang["text"] = dev_lang["text"].apply(clean_text)

    labels = train_lang["label"].tolist()
    majority_class = max(set(labels), key=labels.count)
    minority_class = min(set(labels), key=labels.count)
    majority_samples = train_lang[train_lang["label"] == majority_class]
    minority_samples = train_lang[train_lang["label"] == minority_class]
    augmented_minority = minority_samples.sample(len(majority_samples), replace=True, random_state=42)

    train_combined = pd.concat([majority_samples, minority_samples, augmented_minority])
    train_balanced = train_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    minority_augmented = train_balanced[train_balanced['label'] == minority_class].copy()
    minority_augmented["text"] = minority_augmented["text"].apply(lambda x: augment_text(x, tokenizer))
    train_balanced = pd.concat([train_balanced, minority_augmented])

    dataset_train = TextDataset(train_balanced, tokenizer)
    dataset_dev = TextDataset(dev_lang, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=f"./results_{lang}",
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",  
        save_steps=1000,
        save_total_limit=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        learning_rate=2e-5,
        weight_decay=0.1,
        logging_dir=f"./logs_{lang}",
        logging_steps=500,
        load_best_model_at_end=True,
        report_to="none",
        fp16=True,
        logging_first_step=True,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_dev,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )

    trainer.train()

    os.makedirs(f"./logs_{lang}", exist_ok=True)

    log_history = trainer.state.log_history
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []

    for log in log_history:
        if "loss" in log:
            train_steps.append(log["step"])
            train_losses.append(log["loss"])
        if "eval_loss" in log:
            eval_steps.append(log["step"])
            eval_losses.append(log["eval_loss"])

    plt.figure(figsize=(10, 5))
    plt.plot(train_steps, train_losses, label='Training Loss', linewidth=2)
    plt.plot(eval_steps, eval_losses, label='Validation Loss', marker='o', linewidth=2)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(f"Training vs Validation Loss for {lang}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./logs_{lang}/loss_curve_{lang}.png")
    plt.close()

    print(f"📈 Loss curve saved to ./logs_{lang}/loss_curve_{lang}.png")

    trainer.evaluate()

    predictions = trainer.predict(dataset_dev)
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()
    y_true = dev_lang["label"].tolist()

    thresholds = np.arange(0.3, 0.71, 0.01)
    best_thresh, best_f1 = 0.5, 0

    for t in thresholds:
        preds_thresh = (probs >= t).astype(int)
        macro_f1 = f1_score(y_true, preds_thresh, average='macro', zero_division=1)
        if macro_f1 > best_f1:
            best_thresh, best_f1 = t, macro_f1

    print(f" Best Threshold for {lang}: {best_thresh:.2f} → Macro F1: {best_f1:.4f}")

    y_pred = (probs >= best_thresh).astype(int)
    dev_ids = dev_lang["id"].tolist()
    lang_df = pd.DataFrame({"id": dev_ids, "predicted_label": y_pred})
    all_dev_predictions.append(lang_df)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix for {model_name} ({lang})")
    plt.savefig(f"confusion_matrix_{lang}.png")
    print(f"Confusion matrix saved for {lang}.")

    results = trainer.evaluate()
    from tabulate import tabulate
    print(tabulate([[k, f"{v:.4f}"] for k, v in results.items()], headers=["Metric", "Score"], tablefmt="fancy_grid"))

final_submission = pd.concat(all_dev_predictions, ignore_index=True)
final_submission.to_csv("submission_dev_combined.csv", index=False)
print("\n✅ Combined dev predictions saved as: submission_dev_combined.csv")