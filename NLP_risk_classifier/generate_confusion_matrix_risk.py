import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json

SRC = Path("/content/drive/MyDrive/NLP_risk_classifier")
DATA = SRC / "data"
MODELS = SRC / "models"
RESULTS = SRC / "results" / "risk"

# Load data
df = pd.read_parquet(DATA / "test_split.parquet")

# Load encoder
import pickle
with open(MODELS / "risk_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Load model
from train_roberta_risk import RobertaRisk
num_classes = len(le.classes_)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = RobertaRisk(num_classes)
model.load_state_dict(torch.load(MODELS / "roberta_risk_gpu.pt", map_location=device))
model.to(device)
model.eval()

from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

preds = []
trues = df["risk_label"].tolist()

for text in df["clean_text"].tolist():
    encoded = tokenizer(text, truncation=True, padding="max_length", max_length=96, return_tensors="pt")
    with torch.no_grad():
        out = model(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))
        preds.append(out.argmax(dim=1).cpu().item())

cm = confusion_matrix(trues, preds)

# Save JSON
with open(RESULTS / "confusion_matrix.json", "w") as f:
    json.dump(cm.tolist(), f, indent=4)

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(RESULTS / "confusion_matrix.png")
plt.show()
