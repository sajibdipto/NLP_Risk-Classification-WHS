import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import RobertaTokenizer
from pathlib import Path

from utils import load_pickle
from train_roberta_risk import RobertaRisk

# ---------------- Paths ----------------
BASE = Path("/content/drive/MyDrive/NLP_risk_classifier")
DATA_FILE = BASE / "data/test_split.parquet"
MODEL_DIR = BASE / "models"
OUT_DIR = BASE / "results"
OUT_DIR.mkdir(exist_ok=True)

# ---------------- Load data ----------------
df = pd.read_parquet(DATA_FILE)
df = df.dropna(subset=["risk_label", "clean_text"])

# ---------------- Encoder ----------------
RISK_ENCODER = load_pickle(MODEL_DIR / "risk_encoder.pkl")

# ---------------- Model ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

model = RobertaRisk(len(RISK_ENCODER.classes_))
model.load_state_dict(
    torch.load(MODEL_DIR / "roberta_risk_gpu.pt", map_location=device)
)
model.to(device).eval()

# ---------------- Batched inference ----------------
BATCH = 16
preds = []

for i in range(0, len(df), BATCH):
    texts = df["clean_text"].iloc[i:i+BATCH].tolist()

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64
    ).to(device)

    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
        preds.extend(logits.argmax(dim=1).cpu().tolist())

# ---------------- Confusion matrix ----------------
true = df["risk_label"].tolist()
cm = confusion_matrix(true, preds)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=RISK_ENCODER.classes_
)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.tight_layout()

out_path = OUT_DIR / "confusion_matrix_risk.png"
plt.savefig(out_path, dpi=200)
plt.close()

print("Confusion matrix saved to:")
print(out_path)
