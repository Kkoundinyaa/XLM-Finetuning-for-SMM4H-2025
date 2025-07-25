import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pickle
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

# Load original dev file and prediction output separately
dev_csv = "/users/PAS2836/krishnakb/ondemand/krishna_proj/Task1/dev_data_SMM4H_2025_Task_1.csv"
pred_csv = "submission_dev_combined.csv"  # output from train script

# Load and merge
true_df = pd.read_csv(dev_csv)[["id", "label", "language"]]
pred_df = pd.read_csv(pred_csv)[["id", "predicted_label"]]
df = pd.merge(true_df, pred_df, on="id", how="inner")

# Metrics summary per language
metrics_summary = []

for lang in df['language'].unique():
    lang_df = df[df['language'] == lang]
    y_true = lang_df['label']
    y_pred = lang_df['predicted_label']

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=1)
    rec = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=1)

    metrics_summary.append([lang, acc, prec, rec, f1, f1_macro])

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {lang}")
    os.makedirs("confusion_matrices", exist_ok=True)
    plt.savefig(f"confusion_matrices/confusion_matrix_{lang}.png")
    plt.close()

# Save metric table
metrics_df = pd.DataFrame(metrics_summary, columns=["Language", "Accuracy", "Precision", "Recall", "F1", "F1_macro"])
metrics_df.to_csv("language_metrics.csv", index=False)
print("✅ Saved language-wise metrics to language_metrics.csv")

# Plot loss curves
for lang in df['language'].unique():
    log_path = f"logs_{lang}/log_history.pkl"
    if not os.path.exists(log_path):
        continue

    with open(log_path, "rb") as f:
        log_history = pickle.load(f)

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
    plt.title(f"Loss Curve - {lang}")
    plt.legend()
    plt.grid(True)
    os.makedirs("loss_curves", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"loss_curves/loss_curve_{lang}.png")
    plt.close()

print("✅ Confusion matrices and loss curves saved for all languages.")
