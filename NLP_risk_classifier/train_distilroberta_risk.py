import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# ======================================================
# CONFIG
# ======================================================
DATA_PATH = "/content/drive/MyDrive/NLP_risk_classifier/data/master_data.xlsx"
SHEET_NAME = "MasterData"
TEXT_COL = "WHAT_HAPPENED_ENGLISH"
RISK_COL = "SAFETY_RISK_CATEGORY"

MODEL_NAME = "distilroberta-base"
MAX_LEN = 192
SEED = 42
MIN_SAMPLES = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# CLEAN
# ======================================================
def clean_text(x):
    x = "" if x is None else str(x)
    x = x.strip()
    x = re.sub(r"\s+", " ", x)
    return x

# ======================================================
# LOAD
# ======================================================
df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
df = df[[TEXT_COL, RISK_COL]].copy()
df[TEXT_COL] = df[TEXT_COL].apply(clean_text)
df = df[(df[TEXT_COL].str.len() > 0) & (df[RISK_COL].notna())].reset_index(drop=True)

print("Original labels:", df[RISK_COL].nunique())

# ======================================================
# MERGE RARE LABELS
# ======================================================
counts = df[RISK_COL].value_counts()
rare_labels = counts[counts < MIN_SAMPLES].index.tolist()

print("\nMerging rare labels:", rare_labels)

df[RISK_COL] = df[RISK_COL].apply(
    lambda x: "RARE_OTHER" if x in rare_labels else x
)

print("New label count:", df[RISK_COL].nunique())

# ======================================================
# COARSE GROUP MAP
# ======================================================
GROUP_MAP = {
    "FIRE_EXPLOSION_ENERGY_RELEASE": [
        "Process Safety",
        "Non Process Fire & Explosion",
        "Non-process Fire and Explosion (obs)",
        "Explosives and blasting",
        "Energy release (excl. Electrical)",
    ],
    "LOSS_OF_CONTAINMENT": ["Loss of Containment"],
    "ELECTRICAL": ["Electrical (incl. Arc Flash/Blast)"],
    "GEOTECH_STRUCTURAL": ["Geotechnical Stability"],
    "VEHICLE_MOBILE": [
        "Vehicles & Mobile Equipment",
        "Aviation",
        "Material movement",
        "Tyre handling",
    ],
    "DROPPED_OBJECTS_LIFTING": [
        "Dropped / Falling Object",
        "Lifting",
    ],
    "ENGULFMENT_CRUSHING": [
        "Engulfment / inrush",
        "Entanglement / crushing",
    ],
    "FALLS_HEIGHT": ["Fall from height"],
    "CHEMICAL_HEALTH_EXPOSURE": [
        "Acute Chemical Exposure",
        "Carcinogen Exposure",
        "Occupational Safety",
        "Physical Health",
        "Mental Health",
    ],
    "CONFINED_ATMOSPHERIC": ["Confined Space"],
    "GOVERNANCE_OTHER": [
        "Asset Integrity",
        "Other unspecified",
    ],
    "RARE_GROUP": ["RARE_OTHER"]
}

risk_to_group = {}
for g, labs in GROUP_MAP.items():
    for l in labs:
        risk_to_group[l] = g

df["COARSE_GROUP"] = df[RISK_COL].apply(
    lambda x: risk_to_group.get(x, "RARE_GROUP")
)

print("\nCoarse group distribution:")
print(df["COARSE_GROUP"].value_counts())

# ======================================================
# DATASET CLASS
# ======================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class TextDataset(Dataset):
    def __init__(self, texts, labels, label2id):
        self.texts = texts
        self.labels = labels
        self.label2id = label2id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.label2id[self.labels[idx]])
        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted")
    }

def compute_weights(label_ids, num_labels):
    if len(np.unique(label_ids)) < 2:
        return None
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_labels),
        y=label_ids
    )
    return torch.tensor(weights, dtype=torch.float)

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device)
            )
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ======================================================
# SPLIT
# ======================================================
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=SEED,
    stratify=df["COARSE_GROUP"]
)

# ======================================================
# STAGE 1 TRAIN
# ======================================================
stage1_labels = sorted(df["COARSE_GROUP"].unique())
s1_l2id = {l:i for i,l in enumerate(stage1_labels)}
s1_id2l = {i:l for l,i in s1_l2id.items()}

train_ds = TextDataset(train_df[TEXT_COL].tolist(), train_df["COARSE_GROUP"].tolist(), s1_l2id)
test_ds  = TextDataset(test_df[TEXT_COL].tolist(), test_df["COARSE_GROUP"].tolist(), s1_l2id)

y_train = np.array([s1_l2id[x] for x in train_df["COARSE_GROUP"]])
cw = compute_weights(y_train, len(stage1_labels))

model_stage1 = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(stage1_labels),
    id2label=s1_id2l,
    label2id=s1_l2id
).to(device)

args = TrainingArguments(
    output_dir="stage1",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="no",
    fp16=True,
    seed=SEED,
    report_to="none"
)

trainer1 = WeightedTrainer(
    model=model_stage1,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    class_weights=cw
)

trainer1.train()
print("\nStage-1:", trainer1.evaluate())

# ======================================================
# STAGE 2 TRAIN
# ======================================================
stage2_models = {}
stage2_label_maps = {}

for group in stage1_labels:

    gdf = train_df[train_df["COARSE_GROUP"] == group].copy()

    if gdf[RISK_COL].nunique() <= 1:
        continue

    labels = sorted(gdf[RISK_COL].unique())
    l2id = {l:i for i,l in enumerate(labels)}
    id2l = {i:l for l,i in l2id.items()}

    gtrain, gval = train_test_split(
        gdf,
        test_size=0.2,
        random_state=SEED,
        stratify=gdf[RISK_COL]
    )

    gtrain_ds = TextDataset(gtrain[TEXT_COL].tolist(), gtrain[RISK_COL].tolist(), l2id)
    gval_ds   = TextDataset(gval[TEXT_COL].tolist(), gval[RISK_COL].tolist(), l2id)

    y = np.array([l2id[x] for x in gtrain[RISK_COL]])
    cw2 = compute_weights(y, len(labels))

    model_g = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2l,
        label2id=l2id
    ).to(device)

    trainer_g = WeightedTrainer(
        model=model_g,
        args=args,
        train_dataset=gtrain_ds,
        eval_dataset=gval_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        class_weights=cw2
    )

    print(f"\nTraining Stage-2: {group}")
    trainer_g.train()

    stage2_models[group] = model_g
    stage2_label_maps[group] = l2id

# ======================================================
# FINAL PIPELINE EVALUATION
# ======================================================
print("\nRunning hierarchical evaluation...")

true_labels = []
pred_labels = []

for _, row in test_df.iterrows():

    text = row[TEXT_COL]
    true = row[RISK_COL]

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    ).to(device)

    with torch.no_grad():
        logits1 = model_stage1(**enc).logits
        pred1 = torch.argmax(logits1, dim=1).item()

    coarse = s1_id2l[pred1]

    if coarse not in stage2_models:
        pred = train_df[train_df["COARSE_GROUP"] == coarse][RISK_COL].iloc[0]
    else:
        model2 = stage2_models[coarse]
        l2id = stage2_label_maps[coarse]
        id2l = {v:k for k,v in l2id.items()}

        with torch.no_grad():
            logits2 = model2(**enc).logits
            pred2 = torch.argmax(logits2, dim=1).item()

        pred = id2l[pred2]

    true_labels.append(true)
    pred_labels.append(pred)

print("\nFINAL RESULTS")
print("Accuracy:", accuracy_score(true_labels, pred_labels))
print("Macro F1:", f1_score(true_labels, pred_labels, average="macro"))
print("Weighted F1:", f1_score(true_labels, pred_labels, average="weighted"))

print("\nClassification Report:\n")
print(classification_report(true_labels, pred_labels))

# Save Stage-1
model_stage1.save_pretrained("/content/drive/MyDrive/NLP_risk_classifier/stage1_model")
tokenizer.save_pretrained("/content/drive/MyDrive/NLP_risk_classifier/stage1_model")

# Save Stage-2 models
for group, model in stage2_models.items():
    path = f"/content/drive/MyDrive/NLP_risk_classifier/stage2_{group}"
    model.save_pretrained(path)

print("All models saved successfully.")