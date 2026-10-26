import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

SEED = 42
MODEL_NAME = "csebuetnlp/banglabert"
DATA_FILE = "data/processed/banfakenews_binary_clean.csv"
OUTPUT_DIR = "outputs/banglabert_baseline"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_FILE)

# Use headline + content
df["text"] = df["headline_clean"].astype(str) + " [SEP] " + df["content_clean"].astype(str)

# Stratified split: train 70%, val 15%, test 15%
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=SEED
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=SEED
)

print("Train size:", len(train_df))
print("Validation size:", len(val_df))
print("Test size:", len(test_df))

print("\nTrain label distribution:")
print(train_df["label"].value_counts())

print("\nValidation label distribution:")
print(val_df["label"].value_counts())

print("\nTest label distribution:")
print(test_df["label"].value_counts())

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

train_dataset = NewsDataset(train_df["text"], train_df["label"], tokenizer)
val_dataset = NewsDataset(val_df["text"], val_df["label"], tokenizer)
test_dataset = NewsDataset(test_df["text"], test_df["label"], tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")
    weighted_f1 = f1_score(labels, predictions, average="weighted")
    precision = precision_score(labels, predictions, average="macro")
    recall = recall_score(labels, predictions, average="macro")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": precision,
        "macro_recall": recall
    }

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=100,
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

print("\nStarting training...")
trainer.train()

print("\nEvaluating on validation set...")
val_metrics = trainer.evaluate(eval_dataset=val_dataset)
print(val_metrics)

print("\nEvaluating on test set...")
test_metrics = trainer.evaluate(eval_dataset=test_dataset)
print(test_metrics)

print("\nGenerating test predictions...")
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

print("\nClassification Report:")
print(classification_report(true_labels, preds, target_names=["Fake", "Authentic"]))

print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, preds))

# Save final model and tokenizer
final_model_dir = os.path.join(OUTPUT_DIR, "final_model")
trainer.save_model(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

# Save test predictions
prediction_df = pd.DataFrame({
    "true_label": true_labels,
    "predicted_label": preds
})

prediction_df.to_csv(
    os.path.join(OUTPUT_DIR, "test_predictions.csv"),
    index=False
)

print("\nModel saved to:", final_model_dir)
print("Test predictions saved to:", os.path.join(OUTPUT_DIR, "test_predictions.csv"))