import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Path to the student model directory
# Change this if the model lives in a different folder
MODEL_PATH = "outputs/student-distilled"

# Load tokenizer + model once at startup
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # eval mode


def predict_label_and_probs(text: str):
    """
    Run the student model and return:
      - label: "REAL" or "FAKE"
      - probs: dict like {"REAL": 0.93, "FAKE": 0.07}
    """

    # Tokenize
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    # Inference
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    # Try to infer which index is REAL / FAKE from model config
    real_idx, fake_idx = 0, 1  # sensible default

    id2label = getattr(model.config, "id2label", None)
    if id2label:
        # id2label may be {0: "REAL", 1: "FAKE"} or similar
        temp_real = None
        temp_fake = None
        for k, v in id2label.items():
            name = str(v).upper()
            if "REAL" in name:
                temp_real = int(k)
            if "FAKE" in name:
                temp_fake = int(k)
        if temp_real is not None and temp_fake is not None:
            real_idx, fake_idx = temp_real, temp_fake

    real_prob = float(probs[real_idx])
    fake_prob = float(probs[fake_idx])

    label = "REAL" if real_prob >= fake_prob else "FAKE"

    return label, {"REAL": real_prob, "FAKE": fake_prob}

# ---------- Shared inference helpers for student model ----------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_model(model_dir="outputs/student-distilled", device=None):
    """
    Load the distilled student model + tokenizer for inference.
    Returns: (model, tokenizer, label_map)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # IMPORTANT:
    # Our checkpoint only knows LABEL_0 and LABEL_1.
    # From the NASA test behavior, LABEL_0 behaves like "FAKE" and LABEL_1 like "REAL".
    label_map = {
        0: "FAKE",   # LABEL_0
        1: "REAL",   # LABEL_1
    }

    return model, tokenizer, label_map


def predict_label_and_probs(text, model, tokenizer, label_map):
    """
    Run a single text through the student model and return:
      - label: "REAL" or "FAKE"
      - probs: dict like {"REAL": 0.23, "FAKE": 0.77}
    """
    device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs_tensor = torch.softmax(logits, dim=-1)[0]

    probs = {
        label_map[i]: float(probs_tensor[i].item())
        for i in range(len(probs_tensor))
    }

    # pick label with highest probability
    label = max(probs, key=probs.get)

    return label, probs
