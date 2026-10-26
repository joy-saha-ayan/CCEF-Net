import os
import glob
import json
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = "outputs/banglabert_baseline"
PRED_FILE = os.path.join(BASE_DIR, "test_predictions.csv")

print("=" * 60)
print("BANGLABERT BASELINE RESULT SUMMARY")
print("=" * 60)

if not os.path.exists(PRED_FILE):
    print("Prediction file not found:", PRED_FILE)
else:
    df = pd.read_csv(PRED_FILE)

    true_labels = df["true_label"]
    predicted_labels = df["predicted_label"]

    accuracy = accuracy_score(true_labels, predicted_labels)
    macro_f1 = f1_score(true_labels, predicted_labels, average="macro")
    weighted_f1 = f1_score(true_labels, predicted_labels, average="weighted")
    macro_precision = precision_score(true_labels, predicted_labels, average="macro")
    macro_recall = recall_score(true_labels, predicted_labels, average="macro")

    print("\nOverall Test Metrics:")
    print(f"Accuracy         : {accuracy:.4f}")
    print(f"Macro F1         : {macro_f1:.4f}")
    print(f"Weighted F1      : {weighted_f1:.4f}")
    print(f"Macro Precision  : {macro_precision:.4f}")
    print(f"Macro Recall     : {macro_recall:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        true_labels,
        predicted_labels,
        target_names=["Fake", "Authentic"]
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, predicted_labels))

print("\n" + "=" * 60)
print("BEST VALIDATION METRICS")
print("=" * 60)

checkpoint_dirs = glob.glob(os.path.join(BASE_DIR, "checkpoint-*"))

if checkpoint_dirs:
    latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)
    state_file = os.path.join(latest_checkpoint, "trainer_state.json")

    print("Latest checkpoint:", latest_checkpoint)

    if os.path.exists(state_file):
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)

        log_history = state.get("log_history", [])

        eval_logs = [
            log for log in log_history
            if "eval_macro_f1" in log
        ]

        if eval_logs:
            best_eval = max(eval_logs, key=lambda x: x["eval_macro_f1"])

            print("\nBest validation metrics:")
            for key, value in best_eval.items():
                print(f"{key}: {value}")
        else:
            print("No evaluation metrics found in trainer_state.json")
    else:
        print("trainer_state.json not found.")
else:
    print("No checkpoint folder found.")