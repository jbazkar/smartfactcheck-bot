import os
import math
import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# 1. Load local CSVs and combine
# -----------------------------
DATA_DIR = r"C:\Users\bskrn\datasets\fake_news"  # <-- change if needed

fake_path = os.path.join(DATA_DIR, "Fake.csv")
true_path = os.path.join(DATA_DIR, "True.csv")

print("Loading data from:")
print("  Fake:", fake_path)
print("  True:", true_path)

df_fake = pd.read_csv(fake_path)
df_true = pd.read_csv(true_path)

df_fake["label"] = 0  # 0 = fake
df_true["label"] = 1  # 1 = real

df_all = pd.concat(
    [df_fake[["title", "text", "label"]], df_true[["title", "text", "label"]]],
    ignore_index=True,
)

print("Combined shape:", df_all.shape)
print(df_all["label"].value_counts())

# -----------------------------
# 2. Train / test split
# -----------------------------
train_df, test_df = train_test_split(
    df_all,
    test_size=0.2,
    random_state=42,
    stratify=df_all["label"],
)

print("Train size:", train_df.shape)
print("Test size :", test_df.shape)

# -----------------------------
# 3. Dataset & DataLoader
# -----------------------------
# FAST TEACHER SETTINGS (Option A)
MODEL_NAME = "distilbert-base-uncased"  # smaller & faster than bert-base
MAX_LEN = 128                           # shorter sequences = faster
EPOCHS = 1                              # single pass over data for now
BATCH_SIZE = 8                          # keep moderate for CPU

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class NewsDataset(Dataset):
    def __init__(self, df):
        self.titles = df["title"].tolist()
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        title = self.titles[idx] if isinstance(self.titles[idx], str) else ""
        text = self.texts[idx] if isinstance(self.texts[idx], str) else ""
        combined = title + " " + text

        enc = tokenizer(
            combined,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        return item

train_dataset = NewsDataset(train_df)
test_dataset = NewsDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# -----------------------------
# 4. Model, optimizer, device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# -----------------------------
# 5. Training & evaluation loops
# -----------------------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # light progress logging
        if (step + 1) % 200 == 0:
            print(f"  [train] step {step+1} - avg loss so far: {total_loss / num_batches:.4f}")

    avg_loss = total_loss / max(1, num_batches)
    return avg_loss

def eval_model(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            num_batches += 1

            preds = torch.argmax(logits, dim=-1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(1, num_batches)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, acc, f1

# -----------------------------
# 6. Run training (FAST: 1 epoch)
# -----------------------------
for epoch in range(1, EPOCHS + 1):
    start_time = time.time()
    print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_loss, val_acc, val_f1 = eval_model(model, test_loader, device)
    elapsed = time.time() - start_time

    print(f"Epoch {epoch} done.")
    print(f"  Train loss: {train_loss:.4f}")
    print(f"  Val   loss: {val_loss:.4f}")
    print(f"  Val   acc : {val_acc:.4f}")
    print(f"  Val   f1  : {val_f1:.4f}")
    print(f"  Elapsed   : {elapsed/60:.2f} min")

# -----------------------------
# 7. Save the teacher model
# -----------------------------
save_dir = "outputs/teacher-fast-distilbert"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"\nFast teacher model saved to: {save_dir}")
