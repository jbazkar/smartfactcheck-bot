import os
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

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

# Add numeric labels: 0 = fake, 1 = real
df_fake["label"] = 0
df_true["label"] = 1

# Keep only what we need: title, text, label
df_all = pd.concat([df_fake[["title", "text", "label"]],
                    df_true[["title", "text", "label"]]],
                   ignore_index=True)

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

# Convert to Hugging Face Dataset
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

# -----------------------------
# 3. Tokenizer and preprocessing
# -----------------------------
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(batch):
    # You can experiment with title+text vs text only
    # Here we combine: "[TITLE] ... [TEXT]"
    combined_texts = [
        (t if isinstance(t, str) else "") + " " + (x if isinstance(x, str) else "")
        for t, x in zip(batch["title"], batch["text"])
    ]
    return tokenizer(
        combined_texts,
        truncation=True,
        padding="max_length",
        max_length=256,
    )

train_ds = train_ds.map(preprocess, batched=True)
test_ds = test_ds.map(preprocess, batched=True)

# Remove unused columns (title, text, pandas index, etc.)
cols_to_keep = ["input_ids", "attention_mask", "label"]
train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in cols_to_keep])
test_ds = test_ds.remove_columns([c for c in test_ds.column_names if c not in cols_to_keep])

print("Train columns after tokenization:", train_ds.column_names)
print("Test columns after tokenization :", test_ds.column_names)

# -----------------------------
# 4. Define model & metrics
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1_weighted": f1}

# -----------------------------
# 5. Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="outputs/teacher",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,

    evaluation_strategy="epoch",   # evaluate at end of each epoch
    save_strategy="epoch",         # save at end of each epoch
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
)



# -----------------------------
# 6. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# -----------------------------
# 7. Train and evaluate
# -----------------------------
print("Starting training...")
trainer.train()

print("Evaluating on test set...")
metrics = trainer.evaluate()
print("Test metrics:", metrics)

# -----------------------------
# 8. Save final teacher model
# -----------------------------
save_dir = "outputs/teacher-final"
os.makedirs(save_dir, exist_ok=True)
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Teacher model saved to: {save_dir}")
