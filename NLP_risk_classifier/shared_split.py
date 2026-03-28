import json
import pickle
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =====================================================
# PATHS - FIXED FOR GOOGLE COLAB
# =====================================================
BASE = Path("/content/drive/MyDrive/NLP_risk_classifier")
MASTER_DATA_PATH = BASE / "data" / "master_data.xlsx"
OUTPUT_DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"

OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# CONFIG
# =====================================================
TEXT_COLUMN = "WHAT_HAPPENED_ENGLISH"
RISK_COLUMN = "SAFETY_RISK_CATEGORY"
MIN_SAMPLES = 41
RANDOM_STATE = 42
TEST_SIZE = 0.2

# =====================================================
# LIGHT NORMALIZATION FOR ROBERTA
# =====================================================
def normalize_raw_text(x):
    x = "" if pd.isna(x) else str(x)
    x = x.strip()
    x = re.sub(r"\s+", " ", x)
    return x

# =====================================================
# HEURISTIC FLAGS FROM SUPERVISOR DEFINITIONS
# NOTE: These do NOT overwrite labels.
# They help identify possibly noisy records.
# =====================================================
def make_heuristic_flags(text):
    t = text.lower()

    possible_loc = int(any(k in t for k in [
        "leak", "leaks", "diesel", "fuel", "oil", "overflow",
        "spill", "spillage", "rupture", "ruptured", "rupturing"
    ]))

    possible_process_safety = int(any(k in t for k in [
        "process", "plant", "pipeline", "vessel", "tank",
        "hazardous material", "hazardous materials",
        "release of hazardous", "containment failure"
    ]))

    possible_energy_release = int(any(k in t for k in [
        "unrestrained", "gas cylinder", "gas cylinders",
        "gas bottle", "gas bottles", "incorrect storage",
        "stored incorrectly", "stored unsecured"
    ]))

    possible_acute_chemical = int(any(k in t for k in [
        "safety shower", "chemical exposure", "fume", "fumes",
        "vapour", "vapors", "toxic", "chemical splash"
    ]))

    possible_dropped_object = int(any(k in t for k in [
        "safety glasses", "linen", "laundry", "sharps",
        "sharp edge", "build up", "built up", "dropped object",
        "falling object"
    ]))

    flags = []
    if possible_loc:
        flags.append("possible_LoC")
    if possible_process_safety:
        flags.append("possible_ProcessSafety")
    if possible_energy_release:
        flags.append("possible_EnergyRelease")
    if possible_acute_chemical:
        flags.append("possible_AcuteChemical")
    if possible_dropped_object:
        flags.append("possible_DroppedObject")

    return pd.Series({
        "possible_loc": possible_loc,
        "possible_process_safety": possible_process_safety,
        "possible_energy_release": possible_energy_release,
        "possible_acute_chemical": possible_acute_chemical,
        "possible_dropped_object": possible_dropped_object,
        "heuristic_flag": ";".join(flags) if flags else ""
    })

# =====================================================
# LOAD DATA
# =====================================================
print("\nLoading master dataset...")
print("Expected file:", MASTER_DATA_PATH)

if not MASTER_DATA_PATH.exists():
    raise FileNotFoundError(f"Master data file not found: {MASTER_DATA_PATH}")

df = pd.read_excel(MASTER_DATA_PATH, sheet_name="MasterData")

required_cols = [TEXT_COLUMN, RISK_COLUMN]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df[required_cols].copy()
df = df.dropna(subset=[TEXT_COLUMN, RISK_COLUMN]).reset_index(drop=True)

print(f"Total rows before filtering: {len(df)}")
print(f"Original class count: {df[RISK_COLUMN].nunique()}")

# =====================================================
# RAW TEXT FOR ROBERTA
# =====================================================
df["raw_text"] = df[TEXT_COLUMN].apply(normalize_raw_text)
df = df[df["raw_text"].str.len() > 0].reset_index(drop=True)

# Add heuristic review columns
heuristic_df = df["raw_text"].apply(make_heuristic_flags)
df = pd.concat([df, heuristic_df], axis=1)

# =====================================================
# REMOVE RARE CLASSES (NO MERGING)
# =====================================================
counts = df[RISK_COLUMN].value_counts()
kept_classes = counts[counts >= MIN_SAMPLES].index.tolist()
removed_classes = counts[counts < MIN_SAMPLES].index.tolist()

print(f"\nKeeping classes with >= {MIN_SAMPLES} samples: {len(kept_classes)}")
print("Removed classes:")
print(removed_classes)

df = df[df[RISK_COLUMN].isin(kept_classes)].reset_index(drop=True)

print(f"\nRows after filtering: {len(df)}")
print(f"Remaining class count: {df[RISK_COLUMN].nunique()}")
print("\nClass distribution after filtering:")
print(df[RISK_COLUMN].value_counts())

# =====================================================
# LABEL ENCODING
# =====================================================
risk_le = LabelEncoder()
df["risk_label"] = risk_le.fit_transform(df[RISK_COLUMN])

with open(MODELS_DIR / "risk_encoder.pkl", "wb") as f:
    pickle.dump(risk_le, f)

print("\nSaved:", MODELS_DIR / "risk_encoder.pkl")

# =====================================================
# STRATIFIED TRAIN/TEST SPLIT
# =====================================================
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["risk_label"]
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# =====================================================
# SAVE FILES
# =====================================================
train_path = OUTPUT_DATA_DIR / "train_split.parquet"
test_path = OUTPUT_DATA_DIR / "test_split.parquet"
meta_path = OUTPUT_DATA_DIR / "split_metadata.json"
review_path = OUTPUT_DATA_DIR / "heuristic_review_sample.xlsx"

train_df.to_parquet(train_path, index=False)
test_df.to_parquet(test_path, index=False)

# Save a review file for suspicious rows only
review_df = df[df["heuristic_flag"] != ""].copy()
review_df.to_excel(review_path, index=False)

metadata = {
    "text_column_used": "raw_text",
    "risk_column": RISK_COLUMN,
    "min_samples": MIN_SAMPLES,
    "total_rows_after_filtering": int(len(df)),
    "train_size": int(len(train_df)),
    "test_size": int(len(test_df)),
    "removed_classes": removed_classes,
    "risk_classes": list(risk_le.classes_),
    "num_classes": int(len(risk_le.classes_))
}

with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=4)

print("\nSaved:")
print(" ", train_path)
print(" ", test_path)
print(" ", meta_path)
print(" ", review_path)
print("\nshared_split.py completed successfully.")