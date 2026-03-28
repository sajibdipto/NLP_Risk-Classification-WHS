import torch
import pandas as pd
from transformers import RobertaTokenizer
from pathlib import Path

from utils import load_pickle
from models import RobertaRisk, RobertaMultiTask

# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_DIR = Path("/content/drive/MyDrive/NLP_risk_classifier")
DATA_DIR = BASE_DIR / "data"
CONTROL_DATA_DIR = DATA_DIR / "controls"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

TEST_FILE = DATA_DIR / "test_split.parquet"
OUTPUT_FILE = RESULT_DIR / "combined_evaluation.xlsx"

TASKS = [
    "Control_Fail_(2)",
    "Control_Failed",
    "Control_name",
    "What_Control_Failed",
    "Control_Time",
    "Control_Strength",
]

LABEL_COL_MAP = {
    "Control_Fail_(2)": "Control Fail (2)_label",
    "Control_Failed": "Control Failed_label",
    "Control_name": "Control name_label",
    "What_Control_Failed": "What Control Failed_label",
    "Control_Time": "Control Time_label",
    "Control_Strength": "Control Strength_label",
}

# Ensure output directory
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load encoders
# --------------------------------------------------
RISK_ENCODER = load_pickle(MODEL_DIR / "risk_encoder.pkl")
CONTROL_ENCODERS = {
    t: load_pickle(MODEL_DIR / "controls" / f"{t}_encoder.pkl")
    for t in TASKS
}

# --------------------------------------------------
# Tokenizer + Device
# --------------------------------------------------
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------
# Load Models
# --------------------------------------------------
risk_model = RobertaRisk(len(RISK_ENCODER.classes_))
risk_model.load_state_dict(torch.load(MODEL_DIR / "roberta_risk_gpu.pt", map_location=device))
risk_model.to(device).eval()

multi_model = RobertaMultiTask(
    {t: len(CONTROL_ENCODERS[t].classes_) for t in TASKS},
    TASKS
)
multi_model.load_state_dict(torch.load(MODEL_DIR / "roberta_multitask_gpu.pt", map_location=device))
multi_model.to(device).eval()

# --------------------------------------------------
# Load main test dataset
# --------------------------------------------------
df = pd.read_parquet(TEST_FILE)

# Load each control's true label column
for t in TASKS:
    ctrl_df = pd.read_parquet(CONTROL_DATA_DIR / f"{t}_test.parquet")[["clean_text", LABEL_COL_MAP[t]]]
    df = df.merge(ctrl_df, on="clean_text", how="left")

print("Merged dataset shape:", df.shape)

# --------------------------------------------------
# Batch Inference
# --------------------------------------------------
BATCH = 64
records = []

for start in range(0, len(df), BATCH):
    end = start + BATCH
    batch = df.iloc[start:end]

    texts = batch["clean_text"].tolist()

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=96
    ).to(device)

    # -------- RISK PREDICTION --------
    risk_logits = risk_model(enc["input_ids"], enc["attention_mask"])
    risk_probs = torch.softmax(risk_logits, dim=1)
    risk_pred_idx = risk_probs.argmax(1).cpu().numpy()
    risk_conf = risk_probs.max(1).values.cpu().numpy()

    # -------- CONTROL PREDICTIONS --------
    ctrl_logits = multi_model(enc["input_ids"], enc["attention_mask"])

    ctrl_pred_idx = {t: ctrl_logits[t].argmax(1).cpu().numpy() for t in TASKS}
    ctrl_conf = {
        t: torch.softmax(ctrl_logits[t], dim=1).max(1).values.cpu().numpy()
        for t in TASKS
    }

    # -------- Build Rows --------
    for i in range(len(batch)):
        row = batch.iloc[i]

        rec = {
            "text": row["WHAT_HAPPENED_ENGLISH"],
            "true_risk": row["SAFETY_RISK_CATEGORY"],
            "pred_risk": RISK_ENCODER.classes_[risk_pred_idx[i]],
            "risk_conf": float(risk_conf[i]),
        }

        for t in TASKS:
            true_id = row[LABEL_COL_MAP[t]]
            true_lbl = CONTROL_ENCODERS[t].classes_[true_id]

            pred_lbl = CONTROL_ENCODERS[t].classes_[ctrl_pred_idx[t][i]]

            rec[f"true_{t}"] = true_lbl
            rec[f"pred_{t}"] = pred_lbl
            rec[f"conf_{t}"] = float(ctrl_conf[t][i])

        records.append(rec)

    print(f"Processed {end}/{len(df)} rows...")

# --------------------------------------------------
# Save
# --------------------------------------------------
df_out = pd.DataFrame(records)
df_out.to_excel(OUTPUT_FILE, index=False)

print("\nDONE!")
print("Saved →", OUTPUT_FILE)
