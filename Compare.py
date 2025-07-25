import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tabulate import tabulate

# Load metrics from both models
xlmr_df = pd.read_csv("language_metrics.csv")
xlmv_df = pd.read_csv("language_metrics_xlmv.csv")

xlmr_df = xlmr_df.set_index("Language")
xlmv_df = xlmv_df.set_index("Language")

# Reformat comparison to long format
metrics = ["Accuracy", "Precision", "Recall", "F1", "F1_macro"]
long_form = []

for lang in xlmr_df.index:
    for metric in metrics:
        long_form.append({
            "Language": lang,
            "Metric": metric,
            "XLM-R": xlmr_df.loc[lang, metric],
            "XLM-V": xlmv_df.loc[lang, metric]
        })

comparison_df = pd.DataFrame(long_form)
comparison_df.to_csv("language_metrics_comparison.csv", index=False)

print("\n📊 Reformatted Language-wise Metric Comparison (XLM-R vs XLM-V):\n")
print(tabulate(
    comparison_df,
    headers="keys",
    tablefmt="fancy_grid",
    showindex=False,
    floatfmt=".4f"
))

# Create side-by-side image comparison
unique_langs = comparison_df["Language"].unique()
fig, ax = plt.subplots(figsize=(10, len(unique_langs) * len(metrics) * 0.3))
ax.axis("off")

# Table headers
col_labels = ["Metric", "XLM-R", "XLM-V"]
cell_text = []
row_labels = []

for lang in unique_langs:
    lang_df = comparison_df[comparison_df["Language"] == lang]
    row_labels.extend([lang if i == 0 else "" for i in range(len(metrics))])
    for metric in metrics:
        row = lang_df[lang_df["Metric"] == metric]
        cell_text.append([
            metric,
            f"{row['XLM-R'].values[0]:.3f}",
            f"{row['XLM-V'].values[0]:.3f}"
        ])

# Build table
from matplotlib.table import Table

table = Table(ax, bbox=[0, 0, 1, 1])
nrows, ncols = len(cell_text), len(col_labels)
width, height = 1.0 / (ncols + 1), 1.0 / (nrows + 1)

# Header
for col in range(ncols):
    table.add_cell(0, col, width, height, text=col_labels[col], loc='center', facecolor='#40466e')
    table[(0, col)].get_text().set_color('white')

# Data cells
for i, row in enumerate(cell_text):
    for j, cell in enumerate(row):
        table.add_cell(i+1, j, width, height, text=cell, loc='center', facecolor='white')

# Row separators between languages
for i in range(1, len(row_labels)):
    if row_labels[i] != "" and row_labels[i] != row_labels[i-1]:
        for col in range(ncols):
            table[(i+1, col)].visible_edges = "LRT"

# Row labels
for i, label in enumerate(row_labels):
    if label:
        table.add_cell(i+1, -1, width, height, text=label, loc='right', facecolor='lightgray', edgecolor='black')

ax.add_table(table)
plt.title("XLM-R vs XLM-V: Language-wise Metric Comparison", pad=16)
plt.savefig("xlm_comparison_table.png", bbox_inches='tight')
plt.close()

print("\n✅ Saved clean tabular comparison image as xlm_comparison_table.png")