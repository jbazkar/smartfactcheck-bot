# test_student.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1) Load the distilled student model
MODEL_DIR = "outputs/student-distilled"

print(f"Loading student model from: {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

label_names = {0: "FAKE", 1: "REAL"}

def classify_text(text: str):
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,  # same as training
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    pred_label = int(logits.argmax(dim=-1).cpu().item())
    return pred_label, probs

if __name__ == "__main__":
    while True:
        text = input("\n[Student] Enter a news headline or article (or 'q' to quit):\n> ")
        if text.lower().strip() in ["q", "quit", "exit"]:
            print("Exiting.")
            break

        if not text.strip():
            print("Please enter some text.")
            continue

        label, probs = classify_text(text)
        print(f"\n[Student] Prediction: {label_names[label]}")
        print(f"[Student] Probabilities -> FAKE: {probs[0]:.3f}, REAL: {probs[1]:.3f}\n")
