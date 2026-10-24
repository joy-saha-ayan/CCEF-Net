import pandas as pd

FILE = "data/processed/banfakenews_binary_clean.csv"

df = pd.read_csv(FILE)

print("Rows:", len(df))
print("Columns:", list(df.columns))

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate headline+content:", df.duplicated(subset=["headline", "content"]).sum())

print("\nSample authentic news:")
print(df[df["label"] == 1][["headline_clean", "content_clean"]].head(2))

print("\nSample fake news:")
print(df[df["label"] == 0][["headline_clean", "content_clean"]].head(2))