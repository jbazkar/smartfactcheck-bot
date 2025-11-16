import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# 1. Load data (same split)
# -----------------------------
DATA_DIR = r"C:\Users\bskrn\datasets\fake_news"  # <-- change if needed

fake_path = os.path.join(DATA_DIR, "Fake.csv")
true_path = os.path.join(DATA_DIR, "True.csv")

df_fake = pd.read_csv(fake_path)
df_true = pd.read_csv(true_path)

df_fake["label"] = 0
df_true["label"] = 1

df_all = pd.concat(
    [df_fake[["title", "text", "label"]], df_true[["title", "text", "label"]]],
    ignore_index=True,
)

train_df, test_df = train_test_split(
    df_all,
    test_size=0.2,
    random_state=42,
    stratify=df_all["label"],
)

print("Train size:", train_df.shape)
print("Test size :", test_df.shape)

# -----------------------------
# 2. Load teacher logits
# -----------------------------
logits_dir = "outputs/distillation"

train_teacher_logits = np.load(os.path.join(logits_dir, "train_logits.npy"))
test_teacher_logits = np.load(os.path.join(logits_dir, "test_logits.npy"))

print("Train teacher logits shape:", train_teacher_logits.shape)
print("Test teacher logits shape :", test_teacher_logits.shape)

# -----------------------------
# 3. Student dataset
# -----------------------------
STUDENT_MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 1   # keep fast training: 1 epoch

tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME)

class NewsDistillDataset(Dataset):
    def __init__(self, df, teacher_logits):
        self.titles = df["title"].tolist()
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.teacher_logits = teacher_logits  # numpy [N, 2]

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

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "teacher_logits": torch.tensor(self.teacher_logits[idx], dtype=torch.float),
        }

train_dataset = NewsDistillDataset(train_df.reset_index(drop=True), train_teacher_logits)
test_dataset = NewsDistillDataset(test_df.reset_index(drop=True), test_teacher_logits)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# -----------------------------
# 4. Load student model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

student = AutoModelForSequenceClassification.from_pretrained(
    STUDENT_MODEL_NAME,
    num_labels=2,
)
student.to(device)

# (Optional, small regularization): bump all dropout probs to 0.3
for m in student.modules():
    if isinstance(m, nn.Dropout):
        m.p = 0.3

optimizer = torch.optim.AdamW(student.parameters(), lr=2e-5)

# -----------------------------
# 5. Class weights + KD loss
# -----------------------------
# Compute class weights from training labels to reduce FAKE bias
label_counts = train_df["label"].value_counts()
num_fake = label_counts.get(0, 1)
num_real = label_counts.get(1, 1)
total = num_fake + num_real

# Inverse frequency-style weights
w_fake = total / (2.0 * num_fake)
w_real = total / (2.0 * num_real)
print(f"Class counts -> FAKE: {num_fake}, REAL: {num_real}")
print(f"Class weights -> FAKE: {w_fake:.3f}, REAL: {w_real:.3f}")

class_weights = torch.tensor([w_fake, w_real], dtype=torch.float).to(device)

# Cross-entropy with class weights + label smoothing
ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

def kd_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    """
    Knowledge Distillation loss:
    - T = 4.0: softer teacher distribution
    - alpha = 0.7: more influence from teacher, 0.3 from hard labels
    """
    # Soften teacher distribution & stabilize
    teacher_probs = F.softmax(teacher_logits / T, dim=-1).clamp(min=1e-7, max=1.0)
    student_log_probs = F.log_softmax(student_logits / T, dim=-1)

    # KD term (soft labels)
    loss_kd = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (T * T)

    # Hard label term, but smoothed and class-weighted
    loss_ce = ce_loss_fn(student_logits, labels)

    return alpha * loss_kd + (1.0 - alpha) * loss_ce

# -----------------------------
# 6. Training & evaluation loops
# -----------------------------
def train_student_epoch(model, loader, optimizer, device, T=4.0, alpha=0.7):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for step, batch in enumerate(loader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        teacher_logits = batch["teacher_logits"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        student_logits = outputs.logits

        loss = kd_loss(student_logits, teacher_logits, labels, T=T, alpha=alpha)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (step + 1) % 200 == 0:
            print(f"  [student train] step {step+1} - avg loss: {total_loss / num_batches:.4f}")

    return total_loss / max(1, num_batches)


def eval_student(model, loader, device):
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
            teacher_logits = batch["teacher_logits"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            student_logits = outputs.logits

            # Use same KD setup in eval
            loss = kd_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7)

            preds = torch.argmax(student_logits, dim=-1)

            total_loss += loss.item()
            num_batches += 1

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(1, num_batches)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, acc, f1

# -----------------------------
# 7. Run student training
# -----------------------------
for epoch in range(1, EPOCHS + 1):
    start_time = time.time()
    print(f"\n=== Student Epoch {epoch}/{EPOCHS} ===")
    train_loss = train_student_epoch(student, train_loader, optimizer, device)
    val_loss, val_acc, val_f1 = eval_student(student, test_loader, device)
    elapsed = time.time() - start_time

    print(f"Student Epoch {epoch} done.")
    print(f"  Train loss: {train_loss:.4f}")
    print(f"  Val   loss: {val_loss:.4f}")
    print(f"  Val   acc : {val_acc:.4f}")
    print(f"  Val   f1  : {val_f1:.4f}")
    print(f"  Elapsed   : {elapsed/60:.2f} min")

# -----------------------------
# 8. Save student model
# -----------------------------
save_dir = "outputs/student-distilled"
os.makedirs(save_dir, exist_ok=True)
student.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"\nStudent (distilled) model saved to: {save_dir}")
