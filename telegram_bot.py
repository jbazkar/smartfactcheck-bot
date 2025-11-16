import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from telegram.ext import Updater, MessageHandler, Filters, CommandHandler

# --------- Model setup ----------
MODEL_DIR = "outputs/student-distilled"
MAX_LEN = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

LABEL_MAP = {0: "FAKE", 1: "REAL"}  # 0 = FAKE, 1 = REAL (same as training)


def predict_label(text: str):
    """Run the student model on a single text string and return label + probabilities."""
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    fake_prob = float(probs[0])
    real_prob = float(probs[1])

    label_id = int(probs.argmax())
    label = LABEL_MAP[label_id]

    return label, fake_prob, real_prob


# --------- Telegram handlers ----------
def start(update, context):
    update.message.reply_text(
        "Hi, I'm SmartFactCheckBot 🧠\n\n"
        "Send me a news headline or a short article, "
        "and I'll predict whether it's REAL or FAKE."
    )


def help_cmd(update, context):
    update.message.reply_text(
        "Just send any English news headline or short paragraph.\n\n"
        "Example:\n"
        "\"Trump on Twitter (Dec 29) – Approval rating, Amazon\""
    )


def handle_text(update, context):
    text = (update.message.text or "").strip()
    print("Got message from Telegram:", repr(text))

    if not text:
        update.message.reply_text("Please send a non-empty text.")
        return

    try:
        label, fake_p, real_p = predict_label(text)

        reply = (
            f"🧠 Prediction: *{label}*\n\n"
            f"Probabilities:\n"
            f"• FAKE: `{fake_p:.3f}`\n"
            f"• REAL: `{real_p:.3f}`"
        )
        update.message.reply_text(reply, parse_mode="Markdown")
    except Exception as e:
        # Log any error to the console and send a friendly message
        print("Error while predicting:", repr(e))
        update.message.reply_text(
            "Sorry, something went wrong while analyzing this text. "
            "Please try again."
        )


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN env var not set")

    print("Using token:", token)
    updater = Updater(token, use_context=True)
    print("Bot username from API:", updater.bot.username)

    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_cmd))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_text))

    print("SmartFactCheckBot is running. Press Ctrl+C to stop.")
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
