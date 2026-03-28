import sys
from pathlib import Path
BASE = Path("/content/drive/MyDrive/NLP_risk_classifier")
sys.path.append(str(BASE))

import torch
from transformers import RobertaTokenizer
from utils import load_pickle
from train_roberta_risk import RobertaRisk
from train_roberta_multitask import RobertaMultiTask, TASKS

MODEL_DIR = BASE / "models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load encoders
RISK_ENCODER = load_pickle(MODEL_DIR / "risk_encoder.pkl")
CONTROL_ENCODERS = {
    t: load_pickle(MODEL_DIR / "controls" / f"{t}_encoder.pkl")
    for t in TASKS
}

# Models
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

risk_model = RobertaRisk(len(RISK_ENCODER.classes_))
risk_model.load_state_dict(torch.load(MODEL_DIR / "roberta_risk_gpu.pt", map_location=device))
risk_model.to(device).eval()

multi_model = RobertaMultiTask(
    {t: len(CONTROL_ENCODERS[t].classes_) for t in TASKS},
    TASKS
)
multi_model.load_state_dict(torch.load(MODEL_DIR / "roberta_multitask_gpu.pt", map_location=device))
multi_model.to(device).eval()

# ---- MANUAL INPUT ----
text = input("\nEnter incident description:\n> ")

enc = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=96
).to(device)

with torch.no_grad():
    risk_logits = risk_model(enc["input_ids"], enc["attention_mask"])
    risk_idx = risk_logits.argmax(dim=1).item()

print("\nPredicted Risk:", RISK_ENCODER.classes_[risk_idx])

ctrl_logits = multi_model(enc["input_ids"], enc["attention_mask"])
for t in TASKS:
    idx = ctrl_logits[t].argmax(dim=1).item()
    print(f"{t}: {CONTROL_ENCODERS[t].classes_[idx]}")
