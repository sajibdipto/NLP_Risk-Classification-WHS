import sys
from pathlib import Path

# =============================
# Paths
# =============================
BASE = Path("/content/drive/MyDrive/NLP_risk_classifier")
sys.path.append(str(BASE))

MODEL_PATH = BASE / "models" / "distilroberta_risk.pt"
ENCODER_PATH = BASE / "models" / "risk_encoder.pkl"
DATA_PATH = BASE / "data" / "test_split.parquet"

# =============================
# Imports
# =============================
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import load_pickle

# =============================
# Device
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =============================
# Load encoder
# =============================
risk_encoder = load_pickle(ENCODER_PATH)
num_classes = len(risk_encoder.classes_)
print("Number of classes:", num_classes)

# =============================
# Load tokenizer + model
# =============================
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

model = AutoModelForSequenceClassification.from_pretrained(
    "distilroberta-base",
    num_labels=num_classes
)

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print("✅ Model loaded successfully")

# =========================================================
# OPTION 1 — Manual Testing
# =========================================================

while True:
    text = input("\nEnter incident description (or type 'exit'): ")
    
    if text.lower() == "exit":
        break

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=96
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_idx = torch.argmax(logits, dim=1).item()

    predicted_label = risk_encoder.classes_[pred_idx]

    print("Predicted Risk Category:", predicted_label)

# =========================================================
# OPTION 2 — Quick Batch Test (first 10 rows)
# Uncomment if needed
# =========================================================

"""
df = pd.read_parquet(DATA_PATH).head(10)

texts = df["clean_text"].tolist()

inputs = tokenizer(
    texts,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=96
).to(device)

with torch.no_grad():
    outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

for i, text in enumerate(texts):
    print("\nTEXT:", text)
    print("Predicted:", risk_encoder.classes_[preds[i]])
"""
