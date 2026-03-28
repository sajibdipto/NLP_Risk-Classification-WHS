"""
Creates train/test for 6 control heads.
Fixes:
- merges rare control classes until min_count >= 2
"""

import pandas as pd
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from cleaning import clean_text

# PATHS

BASE_DIR = Path("/content/drive/MyDrive/NLP_risk_classifier")
DATA_DIR = BASE_DIR / "data"
CONTROL_DIR = BASE_DIR / "data" / "controls"
ENCODER_DIR = BASE_DIR / "models" / "controls"

CONTROL_DIR.mkdir(parents=True, exist_ok=True)
ENCODER_DIR.mkdir(parents=True, exist_ok=True)

MASTER_PATH = DATA_DIR / "master_data.xlsx"

CONTROL_COLUMNS = [
    "Control Fail (2)",
    "Control Failed",
    "Control name",
    "What Control Failed",
    "Control Time",
    "Control Strength",
]

MIN_SAMPLES = 2  # Prevent stratified split errors

# LOAD
print("📥 Loading master data...")
df = pd.read_excel(MASTER_PATH)

cols = ["WHAT_HAPPENED_ENGLISH"] + CONTROL_COLUMNS
df = df[cols].copy()

# CLEAN TEXT
print("🧹 Cleaning text...")
df["clean_text"] = df["WHAT_HAPPENED_ENGLISH"].apply(lambda x: " ".join(clean_text(x)))

# NORMALIZE CONTROL COLUMNS

print("🔧 Normalizing control columns...")

for col in CONTROL_COLUMNS:
    df[col] = (
        df[col]
        .fillna("None")
        .astype(str)
        .str.strip()
        .replace("", "None")
    )

# PROCESS EACH CONTROL
from collections import Counter

for col in CONTROL_COLUMNS:
    print(f"\n=== Processing {col} ===")

    counts = df[col].value_counts()

    rare_classes = counts[counts < MIN_SAMPLES].index.tolist()

    if len(rare_classes) > 0:
        print(f"⚠ Merging rare classes into 'None': {rare_classes}")
        df[col] = df[col].apply(lambda x: "None" if x in rare_classes else x)

    # Encode label
    le = LabelEncoder()
    df[col + "_label"] = le.fit_transform(df[col])

    with open(ENCODER_DIR / f"{col.replace(' ', '_')}_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    y = df[col + "_label"]

    # Safe stratified split
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    train_df = df.loc[train_idx, ["clean_text", col + "_label"]]
    test_df = df.loc[test_idx, ["clean_text", col + "_label"]]

    # Save
    train_df.to_parquet(CONTROL_DIR / f"{col.replace(' ', '_')}_train.parquet", index=False)
    test_df.to_parquet(CONTROL_DIR / f"{col.replace(' ', '_')}_test.parquet", index=False)

    print(f"✔ Saved {col.replace(' ', '_')}_train.parquet")
    print(f"✔ Saved {col.replace(' ', '_')}_test.parquet")

print("\n🎉 Done! Control training files generated.")
