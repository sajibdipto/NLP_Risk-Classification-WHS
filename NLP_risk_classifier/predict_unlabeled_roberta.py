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
RESULT_DIR = BASE / "results" / "unlabeled_predictions"

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

# Confidence thresholds
AUTO_THRESHOLD = 0.80
REVIEW_THRESHOLD = 0.60

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# LIGHT TEXT NORMALIZATION
# =====================================================
def normalize_raw_text(x):
    x = "" if pd.isna(x) else str(x)
    x = x.strip()
    x = re.sub(r"\s+", " ", x)
    return x

# =====================================================
# LOAD MODEL + TOKENIZER + LABEL ENCODER
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
df = pd.read_excel(MASTER_DATA_FILE, sheet_name=SHEET_NAME)

required_cols = [TEXT_COL, LABEL_COL]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Keep original row index for traceability
df = df.copy()
df["original_row_id"] = df.index

# Define unlabeled rows:
# - NaN
# - empty string
# - whitespace only
unlabeled_mask = df[LABEL_COL].isna() | (df[LABEL_COL].astype(str).str.strip() == "")
unlabeled_df = df[unlabeled_mask].copy()

print(f"Total rows in master data: {len(df)}")
print(f"Unlabeled rows found: {len(unlabeled_df)}")

if len(unlabeled_df) == 0:
    print("No unlabeled rows found. Exiting.")
    raise SystemExit

unlabeled_df["raw_text"] = unlabeled_df[TEXT_COL].apply(normalize_raw_text)
unlabeled_df = unlabeled_df[unlabeled_df["raw_text"].str.len() > 0].reset_index(drop=True)

print(f"Unlabeled rows after removing empty text: {len(unlabeled_df)}")

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
        item = {k: v.squeeze(0) for k, v in enc.items()}
        return item

dataset = UnlabeledDataset(unlabeled_df, tokenizer, MAX_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# =====================================================
# PREDICTION
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

        # top-2 for analysis
        top2_scores, top2_ids = torch.topk(probs, k=2, dim=1)

        all_pred_ids.extend(pred_ids.cpu().numpy().tolist())
        all_confidences.extend(confs.cpu().numpy().tolist())

        for ids_row, scores_row in zip(top2_ids.cpu().numpy(), top2_scores.cpu().numpy()):
            top2_labels = risk_le.inverse_transform(ids_row)
            all_top2_labels.append(" | ".join(top2_labels))
            all_top2_scores.append(" | ".join([f"{s:.4f}" for s in scores_row]))

# Convert IDs to class names
pred_labels = risk_le.inverse_transform(np.array(all_pred_ids))

# =====================================================
# REVIEW FLAGS
# =====================================================
def confidence_flag(conf):
    if conf >= AUTO_THRESHOLD:
        return "auto_accept"
    elif conf >= REVIEW_THRESHOLD:
        return "manual_review"
    else:
        return "low_confidence"

unlabeled_df["predicted_risk_category"] = pred_labels
unlabeled_df["prediction_confidence"] = np.round(all_confidences, 4)
unlabeled_df["review_flag"] = unlabeled_df["prediction_confidence"].apply(confidence_flag)
unlabeled_df["top2_predicted_labels"] = all_top2_labels
unlabeled_df["top2_scores"] = all_top2_scores

# Optional: add the model prediction into the original label column copy
unlabeled_df["model_filled_risk_category"] = unlabeled_df["predicted_risk_category"]

# =====================================================
# SAVE OUTPUTS
# =====================================================
full_output_path = RESULT_DIR / "unlabeled_predictions_full.xlsx"
csv_output_path = RESULT_DIR / "unlabeled_predictions_full.csv"
review_output_path = RESULT_DIR / "unlabeled_predictions_manual_review.xlsx"
auto_output_path = RESULT_DIR / "unlabeled_predictions_auto_accept.xlsx"

# Save everything
unlabeled_df.to_excel(full_output_path, index=False)
unlabeled_df.to_csv(csv_output_path, index=False)

# Save review subsets
unlabeled_df[unlabeled_df["review_flag"] == "manual_review"].to_excel(review_output_path, index=False)
unlabeled_df[unlabeled_df["review_flag"] == "auto_accept"].to_excel(auto_output_path, index=False)

print("\nSaved files:")
print(" ", full_output_path)
print(" ", csv_output_path)
print(" ", review_output_path)
print(" ", auto_output_path)

print("\nReview flag counts:")
print(unlabeled_df["review_flag"].value_counts())

print("\nSample predictions:")
print(
    unlabeled_df[
        ["original_row_id", TEXT_COL, "predicted_risk_category", "prediction_confidence", "review_flag"]
    ].head(10)
)