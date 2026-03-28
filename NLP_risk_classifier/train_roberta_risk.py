import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

# =====================================================
# PATHS
# =====================================================
BASE = Path("/content/drive/MyDrive/NLP_risk_classifier")
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
RESULT_DIR = BASE / "results" / "risk"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "train_split.parquet"
TEST_FILE = DATA_DIR / "test_split.parquet"
ENCODER_FILE = MODEL_DIR / "risk_encoder.pkl"

# =====================================================
# CONFIG
# =====================================================
MODEL_NAME = "roberta-base"
MAX_LEN = 192
BATCH_SIZE = 8
EPOCHS = 6
LR = 1.5e-5
WEIGHT_DECAY = 0.01
RANDOM_STATE = 42
TEXT_COL = "raw_text"
LABEL_COL = "risk_label"
TARGET_COL = "SAFETY_RISK_CATEGORY"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# SEED
# =====================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_STATE)

# =====================================================
# LOAD FILES
# =====================================================
for p in [TRAIN_FILE, TEST_FILE, ENCODER_FILE]:
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}")

train_df_full = pd.read_parquet(TRAIN_FILE)
test_df = pd.read_parquet(TEST_FILE)

with open(ENCODER_FILE, "rb") as f:
    risk_le = pickle.load(f)

required_cols = [TEXT_COL, LABEL_COL, TARGET_COL]
for c in required_cols:
    if c not in train_df_full.columns:
        raise ValueError(f"Missing column in train split: {c}")
    if c not in test_df.columns:
        raise ValueError(f"Missing column in test split: {c}")

# =====================================================
# CREATE VALIDATION SPLIT FROM TRAIN ONLY
# =====================================================
train_df, val_df = train_test_split(
    train_df_full,
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=train_df_full[LABEL_COL]
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

num_classes = len(risk_le.classes_)

print("Train size:", len(train_df))
print("Validation size:", len(val_df))
print("Test size:", len(test_df))
print("Number of classes:", num_classes)

# =====================================================
# TOKENIZER
# =====================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# =====================================================
# DATASET
# =====================================================
class RiskDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df[TEXT_COL].astype(str).tolist()
        self.labels = df[LABEL_COL].tolist()
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
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = RiskDataset(train_df, tokenizer, MAX_LEN)
val_dataset = RiskDataset(val_df, tokenizer, MAX_LEN)
test_dataset = RiskDataset(test_df, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =====================================================
# MODEL
# =====================================================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_classes
).to(device)

# =====================================================
# CLASS-WEIGHTED LOSS
# =====================================================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df[LABEL_COL]),
    y=train_df[LABEL_COL]
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.15 * total_steps),
    num_training_steps=total_steps
)

use_fp16 = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

# =====================================================
# EVAL FUNCTION
# =====================================================
def run_eval(model, loader):
    model.eval()
    preds, trues = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits, labels)

            total_loss += loss.item()
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()

            preds.extend(batch_preds.tolist())
            trues.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader), 1)

    acc = accuracy_score(trues, preds)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        trues, preds, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        trues, preds, average="weighted", zero_division=0
    )

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "macro_f1": f1_macro,
        "weighted_precision": prec_weighted,
        "weighted_recall": rec_weighted,
        "weighted_f1": f1_weighted,
        "preds": preds,
        "trues": trues
    }

# =====================================================
# TRAINING LOOP
# =====================================================
history = []
best_val_f1 = -1.0
best_model_dir = MODEL_DIR / "roberta_risk_best"

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_fp16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()

        # gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_train_loss += loss.item()

    train_loss = total_train_loss / max(len(train_loader), 1)
    val_metrics = run_eval(model, val_loader)

    row = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_precision": val_metrics["macro_precision"],
        "val_macro_recall": val_metrics["macro_recall"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_weighted_f1": val_metrics["weighted_f1"]
    }
    history.append(row)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | "
        f"val_acc={val_metrics['accuracy']:.4f} | "
        f"val_macro_precision={val_metrics['macro_precision']:.4f} | "
        f"val_macro_f1={val_metrics['macro_f1']:.4f}"
    )

    if val_metrics["macro_f1"] > best_val_f1:
        best_val_f1 = val_metrics["macro_f1"]
        model.save_pretrained(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)
        print(f"Saved best model to {best_model_dir}")

# =====================================================
# LOAD BEST MODEL AND TEST
# =====================================================
best_model = AutoModelForSequenceClassification.from_pretrained(best_model_dir).to(device)
test_metrics = run_eval(best_model, test_loader)

# =====================================================
# SAVE OUTPUTS
# =====================================================
history_df = pd.DataFrame(history)
history_df.to_csv(RESULT_DIR / "roberta_training_history.csv", index=False)

metrics_to_save = {
    "accuracy": test_metrics["accuracy"],
    "macro_precision": test_metrics["macro_precision"],
    "macro_recall": test_metrics["macro_recall"],
    "macro_f1": test_metrics["macro_f1"],
    "weighted_precision": test_metrics["weighted_precision"],
    "weighted_recall": test_metrics["weighted_recall"],
    "weighted_f1": test_metrics["weighted_f1"]
}

with open(RESULT_DIR / "roberta_metrics.json", "w") as f:
    json.dump(metrics_to_save, f, indent=4)

cm = confusion_matrix(test_metrics["trues"], test_metrics["preds"])
pd.DataFrame(
    cm,
    index=risk_le.classes_,
    columns=risk_le.classes_
).to_csv(RESULT_DIR / "roberta_confusion_matrix.csv")

report = classification_report(
    test_metrics["trues"],
    test_metrics["preds"],
    target_names=risk_le.classes_,
    zero_division=0,
    output_dict=True
)
pd.DataFrame(report).transpose().to_csv(RESULT_DIR / "roberta_classification_report.csv")

torch.save(best_model.state_dict(), MODEL_DIR / "roberta_risk_best_state_dict.pt")

print("\nTest metrics:")
print(metrics_to_save)
print("\nSaved files:")
print(" ", RESULT_DIR / "roberta_training_history.csv")
print(" ", RESULT_DIR / "roberta_metrics.json")
print(" ", RESULT_DIR / "roberta_confusion_matrix.csv")
print(" ", RESULT_DIR / "roberta_classification_report.csv")
print(" ", MODEL_DIR / "roberta_risk_best")
print(" ", MODEL_DIR / "roberta_risk_best_state_dict.pt")