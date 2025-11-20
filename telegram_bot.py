import os
import string
from datetime import datetime, timedelta, timezone

import boto3
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from telegram import Update
from telegram.ext import (
    Updater,
    MessageHandler,
    Filters,
    CommandHandler,
    CallbackContext,
)

# --------- Model setup ----------

MODEL_DIR = "outputs/student-distilled"
MAX_LEN = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

# 0 = FAKE, 1 = REAL (based on your REPL tests)
LABEL_MAP = {0: "FAKE", 1: "REAL"}


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


# --------- CloudWatch Metrics Setup ----------

# Let boto3 infer region from the EC2 instance metadata / environment.
cloudwatch = boto3.client("cloudwatch")


def publish_metric(metric_name, value, dimensions=None):
    """
    Publish CloudWatch metrics.

    Behaviour:
    - If `dimensions` is provided (e.g., per ChatId), we publish:
        1) The per-user metric (with the given dimensions)
        2) An aggregated metric with dimension Scope=ALL_USERS
    - If `dimensions` is None, we publish a single global metric.
    """
    try:
        metric_data = []

        if dimensions:
            # Per-user metric, e.g. {Name: "ChatId", Value: "12345"}
            metric_data.append(
                {
                    "MetricName": metric_name,
                    "Value": value,
                    "Unit": "Count",
                    "Dimensions": dimensions,
                }
            )

            # Aggregated metric for all users for the SAME metric name
            metric_data.append(
                {
                    "MetricName": metric_name,
                    "Value": value,
                    "Unit": "Count",
                    "Dimensions": [{"Name": "Scope", "Value": "ALL_USERS"}],
                }
            )
        else:
            # Global metric (no per-user breakdown)
            metric_data.append(
                {
                    "MetricName": metric_name,
                    "Value": value,
                    "Unit": "Count",
                }
            )

        cloudwatch.put_metric_data(
            Namespace="SmartFactCheckBot",
            MetricData=metric_data,
        )

    except Exception as e:
        # Log but do not crash the bot if CloudWatch fails
        print("CloudWatch metric error:", repr(e))


def get_prediction_stats(days: int = 1):
    """
    Fetch the sum of Predictions metric for the last `days` days.
    Uses CloudWatch get_metric_statistics.
    """
    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        resp = cloudwatch.get_metric_statistics(
            Namespace="SmartFactCheckBot",
            MetricName="Predictions",
            StartTime=start_time,
            EndTime=end_time,
            Period=86400,  # 1 day buckets
            Statistics=["Sum"],
        )

        datapoints = resp.get("Datapoints", [])
        if not datapoints:
            return 0.0

        # Sum across all returned datapoints (e.g., multiple days)
        total = sum(dp.get("Sum", 0.0) for dp in datapoints)
        return float(total)
    except Exception as e:
        print("CloudWatch stats error:", repr(e))
        return 0.0


# --------- Input validation helpers ----------


def is_probably_english(text: str) -> bool:
    """
    Very lightweight heuristic:
    - Count alphabetic characters
    - Count ASCII alphabetic characters
    - If most letters are ASCII, treat as 'probably English'
    """
    letters = sum(1 for c in text if c.isalpha())
    ascii_letters = sum(
        1 for c in text if c.isalpha() and c in string.ascii_letters
    )

    if letters == 0:
        return False

    return ascii_letters / letters >= 0.7


def validate_input(text: str):
    """
    Return (ok: bool, error_code: str | None, message: str | None)

    error_code can be used in CloudWatch dimensions, e.g.:
      - "EMPTY"
      - "TOO_SHORT"
      - "NON_TEXTUAL"
      - "TOO_LONG"
      - "NON_ENGLISH"
    """
    cleaned = text.strip()

    if not cleaned:
        return False, "EMPTY", (
            "Please send a non-empty news headline or short paragraph."
        )

    words = cleaned.split()
    if len(words) < 3:
        return False, "TOO_SHORT", (
            "This looks too short to analyze.\n\n"
            "Please send a full news headline or short paragraph "
            "(at least 3–4 words)."
        )

    # Only digits / punctuation
    if not any(c.isalpha() for c in cleaned):
        return False, "NON_TEXTUAL", (
            "I can’t analyze just numbers or symbols.\n\n"
            "Please send an English news headline or short paragraph."
        )

    # Very long input
    if len(cleaned) > 800:
        return False, "TOO_LONG", (
            "That’s a bit long for this model.\n\n"
            "Please send a single headline or short paragraph "
            "(under 800 characters)."
        )

    # Non-English heuristic
    if not is_probably_english(cleaned):
        return False, "NON_ENGLISH", (
            "Right now I only support English text.\n\n"
            "Please send the news headline or summary in English."
        )

    return True, None, None


# --------- Telegram handlers ----------


def start(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id if update.effective_chat else None
    print(f"/start called from chat_id={chat_id}")

    # Metric: someone started / interacted with the bot
    if chat_id is not None:
        publish_metric(
            "Starts",
            1,
            dimensions=[{"Name": "ChatId", "Value": str(chat_id)}],
        )

    update.message.reply_text(
        "Hi, I'm SmartFactCheckBot 🧠\n\n"
        "Send me a news headline or a short article, and I'll predict whether it's REAL or FAKE.\n\n"
        "⚠️ *Disclaimer*\n"
        "I do not access current events.\n"
        "Predictions are based on linguistic patterns from historical datasets (approx. 2015–2020).\n"
        "Always verify important information using trusted, up-to-date sources.",
        parse_mode="Markdown",
    )


def help_cmd(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id if update.effective_chat else None
    print(f"/help called from chat_id={chat_id}")

    if chat_id is not None:
        publish_metric(
            "HelpRequests",
            1,
            dimensions=[{"Name": "ChatId", "Value": str(chat_id)}],
        )

    update.message.reply_text(
        "Just send any English news headline or short paragraph.\n\n"
        "Example:\n"
        "\"Trump on Twitter (Dec 29) – Approval rating, Amazon\""
    )


def handle_text(update: Update, context: CallbackContext):
    text = (update.message.text or "").strip()
    chat_id = update.effective_chat.id if update.effective_chat else None

    print("Got message from Telegram:", repr(text), "from chat_id:", chat_id)

    # Metric: raw incoming messages
    if chat_id is not None:
        publish_metric(
            "Messages",
            1,
            dimensions=[{"Name": "ChatId", "Value": str(chat_id)}],
        )

    # Validate user input before sending it to the model
    is_valid, err_code, err_msg = validate_input(text)
    if not is_valid:
        if chat_id is not None:
            publish_metric(
                "ValidationErrors",
                1,
                dimensions=[
                    {"Name": "ChatId", "Value": str(chat_id)},
                    {"Name": "Reason", "Value": err_code},
                ],
            )
        update.message.reply_text(err_msg)
        return

    try:
        label, fake_p, real_p = predict_label(text)

        # Metrics: successful prediction
        publish_metric("Predictions", 1)
        if chat_id is not None:
            publish_metric(
                "PredictionsPerUser",
                1,
                dimensions=[{"Name": "ChatId", "Value": str(chat_id)}],
            )

        reply = (
            f"🧠 Prediction: *{label}*\n\n"
            f"Probabilities:\n"
            f"• FAKE: `{fake_p:.3f}`\n"
            f"• REAL: `{real_p:.3f}`\n\n"
            "_Note: This model only analyzes English text and older news patterns._"
        )
        update.message.reply_text(reply, parse_mode="Markdown")

    except Exception as e:
        # Metric: prediction error (true internal failure)
        publish_metric("PredictionErrors", 1)
        print("Error while predicting:", repr(e))
        update.message.reply_text(
            "Sorry, something went wrong while analyzing this text. "
            "Please try again."
        )


def stats_cmd(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id if update.effective_chat else None
    print(f"/stats called from chat_id={chat_id}")

    # Optional: metric for stats usage
    if chat_id is not None:
        publish_metric(
            "StatsRequests",
            1,
            dimensions=[{"Name": "ChatId", "Value": str(chat_id)}],
        )

    # Fetch approximate usage stats from CloudWatch
    last_24h = get_prediction_stats(days=1)
    last_7d = get_prediction_stats(days=7)

    reply = (
        "📊 *SmartFactCheckBot Usage Stats*\n\n"
        f"🕒 Last 24 hours: `{int(last_24h)}` predictions\n"
        f"📅 Last 7 days : `{int(last_7d)}` predictions\n\n"
        "_Counts are approximate based on internal CloudWatch metrics._"
    )

    update.message.reply_text(reply, parse_mode="Markdown")


def error_handler(update: Update, context: CallbackContext):
    """Global error handler for unexpected exceptions in handlers."""
    print("Telegram error_handler caught:", repr(context.error))
    publish_metric("TelegramErrors", 1)

    if update and update.effective_message:
        try:
            update.effective_message.reply_text(
                "An unexpected error occurred while handling your request. "
                "Please try again in a moment."
            )
        except Exception:
            # Avoid cascading failures
            pass


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
    dp.add_handler(CommandHandler("stats", stats_cmd))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_text))

    # Global error handler
    dp.add_error_handler(error_handler)

    print("SmartFactCheckBot is running. Press Ctrl+C to stop.")
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
