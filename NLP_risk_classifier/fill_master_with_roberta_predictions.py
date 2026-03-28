import pickle
import re
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
RESULT_DIR = BASE / "results" / "filled_master_predictions"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_DATA_FILE = DATA_DIR / "master_data.xlsx"
MODEL_PATH = MODEL_DIR / "roberta_risk_best"
ENCODER_FILE = MODEL_DIR / "risk_encoder.pkl"

# =====================================================
# CONFIG
# =====================================================
SHEET_NAME = "MasterData"
TEXT_COL = "WHAT_HAPPENED_ENGLISH"
LABEL_COL = "SAFETY_RISK_CATEGORY"
MAX_LEN = 192
BATCH_SIZE = 16

AUTO_THRESHOLD = 0.80
REVIEW_THRESHOLD = 0.60

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# TEXT NORMALIZATION
# =====================================================
def normalize_raw_text(x):
    x = "" if pd.isna(x) else str(x)
    x = x.strip()
    x = re.sub(r"\s+", " ", x)
    return x

def confidence_flag(conf):
    if conf >= AUTO_THRESHOLD:
        return "auto_accept"
    elif conf >= REVIEW_THRESHOLD:
        return "manual_review"
    return "low_confidence"

# =====================================================
# LOAD FILES
# =====================================================
if not MASTER_DATA_FILE.exists():
    raise FileNotFoundError(f"Master dataset not found: {MASTER_DATA_FILE}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Saved model folder not found: {MODEL_PATH}")

if not ENCODER_FILE.exists():
    raise FileNotFoundError(f"Label encoder not found: {ENCODER_FILE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

with open(ENCODER_FILE, "rb") as f:
    risk_le = pickle.load(f)

print("Loaded classes:")
print(list(risk_le.classes_))

# =====================================================
# LOAD MASTER DATA
# =====================================================
df = pd.read_excel(MASTER_DATA_FILE, sheet_name=SHEET_NAME).copy()

required_cols = [TEXT_COL, LABEL_COL]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["original_row_id"] = df.index

# Create output columns
df["PREDICTED_SAFETY_RISK_CATEGORY"] = ""
df["PREDICTION_CONFIDENCE"] = np.nan
df["PREDICTION_REVIEW_FLAG"] = ""
df["TOP2_PREDICTED_LABELS"] = ""
df["TOP2_SCORES"] = ""

# Identify unlabeled rows
unlabeled_mask = df[LABEL_COL].isna() | (df[LABEL_COL].astype(str).str.strip() == "")
unlabeled_df = df[unlabeled_mask].copy()

print(f"Total rows: {len(df)}")
print(f"Unlabeled rows found: {len(unlabeled_df)}")

if len(unlabeled_df) == 0:
    print("No unlabeled rows found. Exiting.")
    raise SystemExit

unlabeled_df["raw_text"] = unlabeled_df[TEXT_COL].apply(normalize_raw_text)
unlabeled_df = unlabeled_df[unlabeled_df["raw_text"].str.len() > 0].copy()

print(f"Unlabeled rows with usable text: {len(unlabeled_df)}")

# =====================================================
# DATASET
# =====================================================
class UnlabeledDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df["raw_text"].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in enc.items()}

dataset = UnlabeledDataset(unlabeled_df, tokenizer, MAX_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# =====================================================
# PREDICT
# =====================================================
all_pred_ids = []
all_confidences = []
all_top2_labels = []
all_top2_scores = []

with torch.no_grad():
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

        confs, pred_ids = torch.max(probs, dim=1)
        top2_scores, top2_ids = torch.topk(probs, k=2, dim=1)

        all_pred_ids.extend(pred_ids.cpu().numpy().tolist())
        all_confidences.extend(confs.cpu().numpy().tolist())

        for ids_row, scores_row in zip(top2_ids.cpu().numpy(), top2_scores.cpu().numpy()):
            top2_labels = risk_le.inverse_transform(ids_row)
            all_top2_labels.append(" | ".join(top2_labels))
            all_top2_scores.append(" | ".join([f"{s:.4f}" for s in scores_row]))

pred_labels = risk_le.inverse_transform(np.array(all_pred_ids))

unlabeled_df["PREDICTED_SAFETY_RISK_CATEGORY"] = pred_labels
unlabeled_df["PREDICTION_CONFIDENCE"] = np.round(all_confidences, 4)
unlabeled_df["PREDICTION_REVIEW_FLAG"] = unlabeled_df["PREDICTION_CONFIDENCE"].apply(confidence_flag)
unlabeled_df["TOP2_PREDICTED_LABELS"] = all_top2_labels
unlabeled_df["TOP2_SCORES"] = all_top2_scores

# =====================================================
# WRITE PREDICTIONS BACK TO FULL MASTER DATAFRAME
# =====================================================
for _, row in unlabeled_df.iterrows():
    idx = row["original_row_id"]
    df.loc[idx, "PREDICTED_SAFETY_RISK_CATEGORY"] = row["PREDICTED_SAFETY_RISK_CATEGORY"]
    df.loc[idx, "PREDICTION_CONFIDENCE"] = row["PREDICTION_CONFIDENCE"]
    df.loc[idx, "PREDICTION_REVIEW_FLAG"] = row["PREDICTION_REVIEW_FLAG"]
    df.loc[idx, "TOP2_PREDICTED_LABELS"] = row["TOP2_PREDICTED_LABELS"]
    df.loc[idx, "TOP2_SCORES"] = row["TOP2_SCORES"]

# Optional: fill a final label column only for high-confidence rows
df["FINAL_RISK_CATEGORY_FILLED"] = df[LABEL_COL]

fill_mask = (
    (df[LABEL_COL].isna() | (df[LABEL_COL].astype(str).str.strip() == "")) &
    (df["PREDICTION_REVIEW_FLAG"] == "auto_accept")
)

df.loc[fill_mask, "FINAL_RISK_CATEGORY_FILLED"] = df.loc[fill_mask, "PREDICTED_SAFETY_RISK_CATEGORY"]

# =====================================================
# SAVE OUTPUTS
# =====================================================
filled_master_xlsx = RESULT_DIR / "master_data_with_roberta_predictions.xlsx"
filled_master_csv = RESULT_DIR / "master_data_with_roberta_predictions.csv"
review_only_xlsx = RESULT_DIR / "rows_for_manual_review.xlsx"

df.to_excel(filled_master_xlsx, index=False)
df.to_csv(filled_master_csv, index=False)

df[df["PREDICTION_REVIEW_FLAG"].isin(["manual_review", "low_confidence"])].to_excel(
    review_only_xlsx, index=False
)

print("\nSaved files:")
print(" ", filled_master_xlsx)
print(" ", filled_master_csv)
print(" ", review_only_xlsx)

print("\nPrediction review flag counts on unlabeled rows:")
print(unlabeled_df["PREDICTION_REVIEW_FLAG"].value_counts())

print("\nNumber of unlabeled rows auto-filled into FINAL_RISK_CATEGORY_FILLED:")
print(fill_mask.sum())