import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import transformers
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                          TrainingArguments, EarlyStoppingCallback)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn import CrossEntropyLoss
import random
import re
import gc

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

def clean_text(text):
    text = re.sub(r'@User|<user>', '', text)
    text = re.sub(r'http\S+|www\.\S+', '[URL]', text)
    emoji_pattern = re.compile("[" u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF" "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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
model_name = "facebook/xlm-v-base"
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

    dataset_train = TextDataset(train_lang, tokenizer)
    dataset_dev = TextDataset(dev_lang, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=f"./results_{lang}_xlmv",
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=4,
        learning_rate=2e-5,
        weight_decay=0.1,
        logging_dir=f"./logs_{lang}_xlmv",
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
    lang_df = pd.DataFrame({"id": dev_ids, "language": lang, "predicted_label": y_pred})
    all_dev_predictions.append(lang_df)

    del model
    torch.cuda.empty_cache()
    gc.collect()

final_submission = pd.concat(all_dev_predictions, ignore_index=True)
final_submission.to_csv("submission_dev_xlm-v.csv", index=False)
print("\n✅ Combined dev predictions saved as: submission_dev_xlm-v.csv")
