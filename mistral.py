import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score
import torch
import re

# Load and clean both train and dev data
train_path = "/users/PAS2836/krishnakb/ondemand/krishna_proj/Task1/train_data_SMM4H_2025_Task_1.csv"
dev_path = "/users/PAS2836/krishnakb/ondemand/krishna_proj/Task1/dev_data_SMM4H_2025_Task_1.csv"
train_df = pd.read_csv(train_path)
dev_df = pd.read_csv(dev_path)

def clean_text(text):
    text = re.sub(r'@User|<user>', '', text)
    text = re.sub(r'http\S+|www\.\S+', '[URL]', text)
    emoji_pattern = re.compile("[" u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF" "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

for df_ in [train_df, dev_df]:
    df_['text'] = df_['text'].astype(str).apply(clean_text)

# Combine both datasets
full_df = pd.concat([train_df, dev_df]).reset_index(drop=True)

# Downsample majority class (Non-ADE)
pos_df = full_df[full_df['label'] == 1]
neg_df = full_df[full_df['label'] == 0].sample(len(pos_df) * 10, random_state=42)
df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# Define language-specific few-shot prompts
prompt_dict = {
    "en": """
Classify the following tweet as either 'ADE' or 'Non-ADE'. Respond with only one of the two labels: ADE or Non-ADE.

Tweet: "I started taking Ibuprofen and now have severe headaches."
Label: ADE

Tweet: "Love this sunny weather! No meds today."
Label: Non-ADE

Tweet: "After taking paracetamol, I developed a rash on my arms."
Label: ADE

Tweet: "I’ve been off meds for a week and feeling fine."
Label: Non-ADE

Tweet: "Took Metformin this morning and had terrible stomach cramps."
Label: ADE

Tweet: "Went for a walk and enjoyed the fresh air."
Label: Non-ADE

Tweet: """ ,

    "ru": """
Определите, содержит ли твит побочное действие (ADE) или нет. Ответьте только 'ADE' или 'Non-ADE'.

Tweet: "Я принял ибупрофен и почувствовал сильную головную боль."
Label: ADE

Tweet: "Просто пошёл на прогулку, никаких лекарств."
Label: Non-ADE

Tweet: "После парацетамола появилась сыпь на коже."
Label: ADE

Tweet: "Пью витамины, чувствую себя отлично."
Label: Non-ADE

Tweet: "Принял аспирин, началась тошнота и головокружение."
Label: ADE

Tweet: "Смотрю фильм и отдыхаю."
Label: Non-ADE

Tweet: """,

    "de": """
Klassifiziere, ob der Tweet ein Nebenwirkung (ADE) enthält oder nicht. Antworte mit 'ADE' oder 'Non-ADE'.

Tweet: "Nach der Einnahme von Ibuprofen hatte ich starke Kopfschmerzen."
Label: ADE

Tweet: "Ich genieße das Wetter und nehme keine Medikamente."
Label: Non-ADE

Tweet: "Ich habe Paracetamol genommen und einen Ausschlag bekommen."
Label: ADE

Tweet: "Fühle mich heute ohne Medikamente sehr gut."
Label: Non-ADE

Tweet: "Aspirin hat mir Übelkeit verursacht."
Label: ADE

Tweet: "Spaziergang im Park heute."
Label: Non-ADE

Tweet: """,

    "fr": """
Classifiez si le tweet contient un effet indésirable (ADE) ou non. Répondez uniquement par 'ADE' ou 'Non-ADE'.

Tweet: "J'ai pris de l'ibuprofène et j'ai eu une forte migraine."
Label: ADE

Tweet: "Je suis allé me promener, pas de médicaments aujourd'hui."
Label: Non-ADE

Tweet: "Après avoir pris du paracétamol, j'ai eu une éruption cutanée."
Label: ADE

Tweet: "Aucun symptôme aujourd'hui, je vais bien."
Label: Non-ADE

Tweet: "La mirtazapine m'a donné des jambes lourdes."
Label: ADE

Tweet: "Je regarde un film tranquille."
Label: Non-ADE

Tweet: """
}

# Load Zephyr model
model_name = "HuggingFaceH4/zephyr-7b-alpha"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
model.eval()

# Predict using model.generate()
y_true, y_pred = [], []
device = model.device

with torch.no_grad():
    for i, row in df.iterrows():
        lang = row['language'] if row['language'] in prompt_dict else 'en'
        prompt = prompt_dict[lang] + row['text'] + "\nLabel:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        output = model.generate(**inputs, max_new_tokens=10)
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        predicted = decoded[len(prompt):].strip().lower()

        if 'non' in predicted and 'ade' in predicted:
            y_pred.append(0)
        elif 'ade' in predicted:
            y_pred.append(1)
        else:
            y_pred.append(0)

        y_true.append(row['label'])

# Metrics
acc = accuracy_score(y_true, y_pred)
f1_bin = f1_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average='macro')

print("\n📊 Evaluation Results with Zephyr-7B (Per-Language Prompting, Full Dataset):")
print(f"Accuracy     : {acc:.4f}")
print(f"F1 Score     : {f1_bin:.4f}")
print(f"F1 Macro     : {f1_macro:.4f}")