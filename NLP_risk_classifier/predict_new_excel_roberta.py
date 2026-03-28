# predict_new_excel_roberta.py

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =====================================================
# PATHS
# =====================================================
BASE = Path("/content/drive/MyDrive/NLP_risk_classifier")
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"

# Your new file
INPUT_FILE = DATA_DIR / "2026.03.19 Events Inc. What Happened for LLM.xlsx"
OUTPUT_FILE = DATA_DIR / "2026.03.19 Events Inc. What Happened for LLM_predicted.xlsx"

# Saved model + encoder
MODEL_PATH = MODEL_DIR / "roberta_risk_best"
ENCODER_FILE = MODEL_DIR / "risk_encoder.pkl"

# =====================================================
# CONFIG
# =====================================================
MODEL_NAME = "roberta-base"
TEXT_COL = "WHAT_HAPPENED_ENGLISH"
PRED_COL = "SAFETY_RISK_CATEGORY"
CONF_COL = "prediction_confidence"
FLAG_COL = "review_flag"

MAX_LEN = 160
BATCH_SIZE = 32

# review thresholds
AUTO_ACCEPT_THRESHOLD = 0.85
LOW_CONF_THRESHOLD = 0.60

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# CHECK FILES
# =====================================================
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Saved model folder not found: {MODEL_PATH}")

if not ENCODER_FILE.exists():
    raise FileNotFoundError(f"Risk encoder not found: {ENCODER_FILE}")

# =====================================================
# LOAD ENCODER + TOKENIZER + MODEL
# =====================================================
with open(ENCODER_FILE, "rb") as f:
    risk_le = pickle.load(f)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

print("Loaded model from:", MODEL_PATH)
print("Number of classes:", len(risk_le.classes_))

# =====================================================
# LOAD EXCEL
# =====================================================
df = pd.read_excel(INPUT_FILE, sheet_name="MasterData")

if TEXT_COL not in df.columns:
    raise ValueError(f"Column '{TEXT_COL}' not found in file. Available columns: {df.columns.tolist()}")

print("Input rows:", len(df))

# Keep original copy
df_out = df.copy()

# =====================================================
# PREPARE TEXTS TO PREDICT
# Only predict rows where WHAT_HAPPENED_ENGLISH exists
# =====================================================
predict_mask = df_out[TEXT_COL].notna() & (df_out[TEXT_COL].astype(str).str.strip() != "")
predict_df = df_out.loc[predict_mask].copy().reset_index()

print("Rows to predict:", len(predict_df))

if len(predict_df) == 0:
    raise ValueError("No valid text found in WHAT_HAPPENED_ENGLISH column.")

# =====================================================
# DATASET
# =====================================================
class PredictDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        return item

dataset = PredictDataset(
    texts=predict_df[TEXT_COL].astype(str).tolist(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# =====================================================
# PREDICTION
# =====================================================
all_preds = []
all_confidences = []

with torch.no_grad():
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        probs = torch.softmax(logits, dim=1)
        confs, preds = torch.max(probs, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_confidences.extend(confs.cpu().numpy().tolist())

# convert labels back
predicted_labels = risk_le.inverse_transform(all_preds)

# =====================================================
# REVIEW FLAG LOGIC
# =====================================================
def get_review_flag(conf):
    if conf >= AUTO_ACCEPT_THRESHOLD:
        return "auto_accept"
    elif conf >= LOW_CONF_THRESHOLD:
        return "manual_review"
    else:
        return "low_confidence"

review_flags = [get_review_flag(x) for x in all_confidences]

# =====================================================
# WRITE BACK TO ORIGINAL DATAFRAME
# =====================================================
df_out[PRED_COL] = df_out.get(PRED_COL, np.nan)
df_out[CONF_COL] = np.nan
df_out[FLAG_COL] = np.nan

for i, original_idx in enumerate(predict_df["index"]):
    df_out.at[original_idx, PRED_COL] = predicted_labels[i]
    df_out.at[original_idx, CONF_COL] = round(float(all_confidences[i]), 4)
    df_out.at[original_idx, FLAG_COL] = review_flags[i]

# =====================================================
# SAVE OUTPUT
# =====================================================
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    df_out.to_excel(writer, sheet_name="MasterData", index=False)

print("\nSaved predicted file to:")
print(OUTPUT_FILE)

# =====================================================
# SUMMARY
# =====================================================
summary = pd.Series(review_flags).value_counts(dropna=False)
print("\nPrediction flag summary:")
print(summary)

print("\nTop predicted categories:")
print(pd.Series(predicted_labels).value_counts().head(20))