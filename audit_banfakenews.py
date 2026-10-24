import os
import glob
import pandas as pd

DATA_DIR = "data/raw/banfakenews"
OUTPUT_DIR = "outputs/reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)

csv_files = glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)

print("=" * 60)
print("BANFAKENEWS-2.0 DATA AUDIT")
print("=" * 60)

if not csv_files:
    print("No CSV files found. Check the dataset folder path.")

all_reports = []

for file_path in csv_files:
    print("\n" + "-" * 60)
    print("FILE:", file_path)

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print("Could not read CSV:", e)
        continue

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Column names:", list(df.columns))

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    # Try to detect label column
    possible_label_cols = [
        "label", "Label", "labels", "class", "Class",
        "category", "Category", "fake", "is_fake"
    ]

    label_col = None
    for col in possible_label_cols:
        if col in df.columns:
            label_col = col
            break

    if label_col:
        print("\nLabel column detected:", label_col)
        print(df[label_col].value_counts())
    else:
        print("\nNo obvious label column found.")

    report = {
        "file": file_path,
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rows": df.duplicated().sum(),
        "label_column": label_col,
        "missing_total": int(df.isnull().sum().sum())
    }

    all_reports.append(report)

if all_reports:
    report_df = pd.DataFrame(all_reports)
    report_path = os.path.join(OUTPUT_DIR, "banfakenews_audit_report.csv")
    report_df.to_csv(report_path, index=False)
    print("\nAudit report saved to:", report_path)