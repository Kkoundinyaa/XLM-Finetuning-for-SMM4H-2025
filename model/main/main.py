import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer, 
                          TrainingArguments, AutoModel, EarlyStoppingCallback)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from torch.nn import CrossEntropyLoss
from collections import Counter
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import shapiro, probplot
import pickle

# --- Data Augmentation Function ---
def augment_text(text, tokenizer, mask_prob=0.15):
    tokens = tokenizer.tokenize(text)
    num_masks = max(1, int(len(tokens) * mask_prob))
    masked_indices = random.sample(range(len(tokens)), num_masks)
    for idx in masked_indices:
        tokens[idx] = "[MASK]"
    return tokenizer.convert_tokens_to_string(tokens)

# --- Load Data ---
train_file = "/users/PAS2836/krishnakb/ondemand/krishna_proj/train_data_SMM4H_2025_Task_1.csv"
dev_file = "/users/PAS2836/krishnakb/ondemand/krishna_proj/dev_data_SMM4H_2025_Task_1.csv"
train_df = pd.read_csv(train_file)
dev_df = pd.read_csv(dev_file)

# --- Balance the Data ---
labels = train_df["label"].tolist()
class_counts = Counter(labels)
print(f"Original Class Distribution: {class_counts}")

majority_class = max(class_counts, key=class_counts.get)
minority_class = min(class_counts, key=class_counts.get)
majority_samples = train_df[train_df["label"] == majority_class]
minority_samples = train_df[train_df["label"] == minority_class]
num_extra_samples = len(majority_samples) - len(minority_samples)
augmented_minority_samples = minority_samples.sample(n=num_extra_samples, replace=True, random_state=42)
train_df_balanced = pd.concat([majority_samples, minority_samples, augmented_minority_samples]).sample(frac=1, random_state=42)
print(f"New Class Distribution: {Counter(train_df_balanced['label'])}")

# --- Correlation Analysis ---
correlation_matrix = train_df_balanced.select_dtypes(include=np.number).corr()
print("Correlation Matrix (Before Filtering): \n", correlation_matrix)
numerical_train_df_balanced = train_df_balanced.select_dtypes(include=[np.number])

model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Augment text for the minority class
augmented_minority_samples = train_df_balanced[train_df_balanced["label"] == minority_class].copy()
augmented_minority_samples["text"] = augmented_minority_samples["text"].apply(lambda x: augment_text(x, tokenizer))
train_df_balanced = pd.concat([train_df_balanced, augmented_minority_samples])

correlation_matrix = numerical_train_df_balanced.corr()
correlation_matrix.to_csv("correlation_matrix.csv")

# --- Data Transformations ---
train_df_balanced["text_length_log"] = np.log1p(train_df_balanced["text"].apply(len))
train_df_balanced["text_length_sqrt"] = np.sqrt(train_df_balanced["text"].apply(len))
train_df_balanced["text_length_cbrt"] = np.cbrt(train_df_balanced["text"].apply(len))
stat, p = shapiro(train_df_balanced["text_length_log"])
print(f"Shapiro-Wilk Test for Training Data: Stat = {stat}, p = {p}")

plt.figure(figsize=(10, 6))
sns.kdeplot(train_df_balanced["text_length_log"], shade=True, label="Log Transformed")
sns.kdeplot(train_df_balanced["text_length_sqrt"], shade=True, label="Sqrt Transformed")
sns.kdeplot(train_df_balanced["text_length_cbrt"], shade=True, label="Cbrt Transformed")
plt.title("Training Data: Before and After Transformation")
plt.legend()
plt.savefig("kde_transformed_plot.png")

probplot(train_df_balanced["text_length_log"], dist="norm", plot=plt)
plt.savefig("qq_plot_log_text_length.png")

# --- Dataset Class ---
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

# --- Tokenizer & Datasets ---
model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, num_labels=2)
base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
dataset_train = TextDataset(train_df_balanced, tokenizer)
dataset_dev = TextDataset(dev_df, tokenizer)

# --- Enhanced Metrics for Contest and Insight ---
def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    preds = np.argmax(eval_pred.predictions, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average='macro', zero_division=1),
        "recall": recall_score(labels, preds, average='macro', zero_division=1),
        "f1_macro": f1_score(labels, preds, average='macro', zero_division=1),   # Contest metric
        "f1": f1_score(labels, preds, average='binary', zero_division=1),
    }

class WeightedThresholdTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
    
    # Set class weights: [negative, positive]
        class_weights = torch.tensor([1.0, 2.0]).to(logits.device)  # tune these
        loss_fct = CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.1,  # Increased for regularization
    logging_dir="./logs",
    logging_steps=200,
    load_best_model_at_end=True,
    report_to="none",
    fp16=True,
)

# --- Trainer with EarlyStopping ---
trainer = WeightedThresholdTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_dev,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# --- Train & Evaluate ---
trainer.train()
trainer.evaluate()

# --- Plot Training & Validation Loss ---
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
plt.plot(train_steps, train_losses, label='Training Loss')
plt.plot(eval_steps, eval_losses, label='Validation Loss', marker='o')
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.savefig("loss_curve.png")
plt.show()

# --- Saving Predictions & Plotting Confusion Matrix ---
predictions = trainer.predict(dataset_dev)
y_pred = predictions.predictions.argmax(axis=1)
y_true = dev_df["label"].tolist()

with open("predictions.pkl", "wb") as f:
    pickle.dump({"y_true": y_true, "y_pred": y_pred}, f)

with open("predictions.pkl", "rb") as f:
    data = pickle.load(f)

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Negative", "Positive"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved as confusion_matrix.png")

plot_confusion_matrix(y_true, y_pred, model_name)

from tabulate import tabulate

results = trainer.evaluate()
metrics_table = [[k, f"{v:.4f}"] for k, v in results.items()]
print("\n🧾 Final Evaluation Metrics:\n")
print(tabulate(metrics_table, headers=["Metric", "Score"], tablefmt="fancy_grid"))
