import json
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import RobertaTokenizer

from utils import ensure_dir, load_pickle
from models import RobertaMultiTask

BASE = Path("/content/drive/MyDrive/NLP_risk_classifier")
DATA_DIR = BASE / "data" / "controls"
MODEL_DIR = BASE / "models"
RESULT_DIR = BASE / "results" / "controls"

ensure_dir(MODEL_DIR)
ensure_dir(RESULT_DIR)

TASKS = [
    "Control_Fail_(2)", "Control_Failed", "Control_name",
    "What_Control_Failed", "Control_Time", "Control_Strength"
]

ENCODERS = {
    t: load_pickle(MODEL_DIR / "controls" / f"{t}_encoder.pkl")
    for t in TASKS
}

NUM_CLASSES = {t: len(ENCODERS[t].classes_) for t in TASKS}

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiTaskDataset(Dataset):
    def __init__(self, dfs):
        base = next(iter(dfs.values()))
        self.texts = base["clean_text"].tolist()

        self.labels = {
            t: dfs[t][[c for c in dfs[t].columns if c.endswith("_label")][0]].tolist()
            for t in TASKS
        }

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=96,
            return_tensors="pt"
        )
        item = {k: v.squeeze() for k, v in enc.items()}
        for t in TASKS:
            item[t] = torch.tensor(self.labels[t][idx])
        return item


if __name__ == "__main__":
    print("Using device:", device)

    train_dfs = {t: pd.read_parquet(DATA_DIR / f"{t}_train.parquet") for t in TASKS}
    test_dfs = {t: pd.read_parquet(DATA_DIR / f"{t}_test.parquet") for t in TASKS}

    train_loader = DataLoader(MultiTaskDataset(train_dfs), batch_size=16, shuffle=True)
    test_loader = DataLoader(MultiTaskDataset(test_dfs), batch_size=16)

    model = RobertaMultiTask(NUM_CLASSES, TASKS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    EPOCHS = 3
    epoch_logs = []

    for ep in range(EPOCHS):
        model.train()
        losses = []

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, mask)
            loss = sum(loss_fn(outputs[t], batch[t].to(device)) for t in TASKS)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg = sum(losses) / len(losses)
        epoch_logs.append({"epoch": ep+1, "loss": avg})
        print(f"Epoch {ep+1} — Loss: {avg:.4f}")

    pd.DataFrame(epoch_logs).to_excel(RESULT_DIR / "epoch_logs.xlsx", index=False)

    # Save model
    torch.save(model.state_dict(), MODEL_DIR / "roberta_multitask_gpu.pt")
    print("Saved → roberta_multitask_gpu.pt")
