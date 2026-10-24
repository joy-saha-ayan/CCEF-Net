import os
import re
import pandas as pd

INPUT_FILE = "data/raw/banfakenews/All60Kdataset.csv"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "banfakenews_binary_clean.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)

print("Original rows:", len(df))
print("Original columns:", list(df.columns))

# Keep only required columns
df = df[["Headline", "Content", "Label"]].copy()

# Rename columns
df = df.rename(columns={
    "Headline": "headline",
    "Content": "content",
    "Label": "label"
})

# Remove exact duplicates
before_dedup = len(df)
df = df.drop_duplicates(subset=["headline", "content"]).reset_index(drop=True)
after_dedup = len(df)

print(f"Removed duplicate rows: {before_dedup - after_dedup}")
print("Rows after deduplication:", after_dedup)

# Remove null or empty content
df = df.dropna(subset=["headline", "content"])
df = df[df["content"].str.strip().str.len() > 0]
df = df[df["headline"].str.strip().str.len() > 0]

print("Rows after removing empty text:", len(df))

# Basic text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text

print("Cleaning text...")
df["headline_clean"] = df["headline"].apply(clean_text)
df["content_clean"] = df["content"].apply(clean_text)

# Remove rows that became empty after cleaning
df = df[df["headline_clean"].str.len() > 0]
df = df[df["content_clean"].str.len() > 0]

print("Rows after text cleaning:", len(df))

# Check label distribution
print("\nLabel distribution:")
print(df["label"].value_counts())

# Save final dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\nClean dataset saved to:", OUTPUT_FILE)
print("Final rows:", len(df))