"""
inference_roberta.py
Runs full inference:
  1. Roberta Risk Classification
  2. Roberta Multi-task Controls Classification
"""

import torch
import pandas as pd
from pathlib import Path
import json
import numpy as np

from transformers import RobertaTokenizer
from cleaning import clean_text

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------
ROOT = Path("/content/drive/MyDrive/NLP_risk_classifier")

MODEL_DIR = ROOT / "models"
RISK_MODEL_PATH = MODEL_DIR / "roberta_risk_gpu.pt"
MULTITASK_MODEL_PATH = MODEL_DIR / "roberta_multitask_gpu.pt"

RISK_ENCODER_PATH = MODEL_DIR / "risk_encoder.pkl"
CONTROL_ENCODERS_DIR = MODEL_DIR / "controls"


# -------------------------------------------------------
# LOAD ENCODERS
# -------------------------------------------------------
import pickle

with open(RISK_ENCODER_PATH, "rb") as f:
    RISK_ENCODER = pickle.load(f)

CONTROL_COLUMNS = [
    "Control_Fail_(2)",
    "Control_Failed",
    "Control_name",
    "What_Control_Failed",
    "Control_Time",
    "Control_Strength"
]

CONTROL_ENCODERS = {
    col: pickle.load(open(CONTROL_ENCODERS_DIR / f"{col}_encoder.pkl", "rb"))
    for col in CONTROL_COLUMNS
}


# -------------------------------------------------------
# LOAD TOKENIZER
# -------------------------------------------------------
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


# -------------------------------------------------------
# MODEL DEFINITIONS (same as training)
# -------------------------------------------------------
from torch import nn
from transformers import RobertaModel


class RobertaRisk(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        pooled = self.roberta(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return self.classifier(pooled)


class RobertaMultiTask(nn.Module):
    def __init__(self, num_classes_dict):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.classifiers = nn.ModuleDict({
            col: nn.Linear(768, num_classes_dict[col])
            for col in CONTROL_COLUMNS
        })

    def forward(self, input_ids, attention_mask):
        pooled = self.roberta(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return {col: self.classifiers[col](pooled) for col in CONTROL_COLUMNS}


# -------------------------------------------------------
# LOAD MODELS
# -------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Risk model
risk_model = RobertaRisk(len(RISK_ENCODER.classes_))
risk_model.load_state_dict(torch.load(RISK_MODEL_PATH, map_location=device))
risk_model.to(device)
risk_model.eval()

# Multi-task model
num_classes_dict = {col: len(CONTROL_ENCODERS[col].classes_) for col in CONTROL_COLUMNS}

multitask_model = RobertaMultiTask(num_classes_dict)
multitask_model.load_state_dict(torch.load(MULTITASK_MODEL_PATH, map_location=device))
multitask_model.to(device)
multitask_model.eval()


# -------------------------------------------------------
# INFERENCE FUNCTION
# -------------------------------------------------------
def predict(text):

    # Clean text using your cleaning.py logic
    tokens = clean_text(text)
    clean_input = " ".join(tokens)

    encoded = tokenizer(
        clean_input,
        truncation=True,
        padding="max_length",
        max_length=96,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device)

    # ---- Risk Prediction ----
    with torch.no_grad():
        risk_logits = risk_model(input_ids, mask)
        risk_probs = torch.softmax(risk_logits, dim=1)[0]
        risk_idx = risk_probs.argmax().item()
        predicted_risk = RISK_ENCODER.inverse_transform([risk_idx])[0]

    # ---- Controls Prediction ----
    controls_output = {}

    with torch.no_grad():
        control_logits = multitask_model(input_ids, mask)

        for col in CONTROL_COLUMNS:
            logits = control_logits[col][0]
            probs = torch.softmax(logits, dim=0)
            idx = probs.argmax().item()

            label = CONTROL_ENCODERS[col].inverse_transform([idx])[0]
            conf = float(probs[idx].item())

            controls_output[col] = {
                "prediction": label,
                "confidence": round(conf, 4)
            }

    # ---- Combine Everything ----
    return {
        "Risk Category": {
            "prediction": predicted_risk,
            "confidence": float(risk_probs[risk_idx])
        },
        "Controls": controls_output
    }


# -------------------------------------------------------
# MANUAL TEST
# -------------------------------------------------------
if __name__ == "__main__":
    sample_text = input("\nEnter incident description:\n> ")

    output = predict(sample_text)

    print("\n===== FINAL OUTPUT =====")
    print(json.dumps(output, indent=4))
