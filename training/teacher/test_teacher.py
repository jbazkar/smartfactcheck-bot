import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1) Load the trained teacher model
MODEL_DIR = "outputs/teacher-fast-distilbert"

print(f"Loading teacher model from: {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

label_names = {0: "FAKE", 1: "REAL"}

def classify_text(text: str):
    # Prepare input for the model
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,  # same as training
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # Disable gradients for inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    pred_label = int(logits.argmax(dim=-1).cpu().item())
    return pred_label, probs

if __name__ == "__main__":
    while True:
        text = input("\nEnter a news headline or article (or 'q' to quit):\n> ")
        if text.lower().strip() in ["q", "quit", "exit"]:
            print("Exiting.")
            break

        if not text.strip():
            print("Please enter some text.")
            continue

        label, probs = classify_text(text)
        print(f"\nPrediction: {label_names[label]}")
        print(f"Probabilities -> FAKE: {probs[0]:.3f}, REAL: {probs[1]:.3f}\n")
