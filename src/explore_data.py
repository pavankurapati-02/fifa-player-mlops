
import pandas as pd
from pathlib import Path
from fifa_churn.modeling import AGE_THRESHOLD, TARGET_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fifa_players.csv"
REPORT_PATH = PROJECT_ROOT / "reports"

REPORT_PATH.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load the dataset
# --------------------------------------------------

print("\n" + "=" * 60)
print("FIFA PLAYER DATASET EXPLORATION")
print("=" * 60)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n"
        "Please place fifa_players.csv inside data/raw/"
    )

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")


# --------------------------------------------------
# 3. Dataset dimensions
# --------------------------------------------------

print("\n1. DATASET SHAPE")
print("-" * 40)

print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")


# --------------------------------------------------
# 4. Display first five records
# --------------------------------------------------

print("\n2. FIRST FIVE ROWS")
print("-" * 40)

print(df.head())


# --------------------------------------------------
# 5. Display column names
# --------------------------------------------------

print("\n3. COLUMN NAMES")
print("-" * 40)

for index, column in enumerate(df.columns, start=1):
    print(f"{index}. {column}")


# --------------------------------------------------
# 6. Check data types
# --------------------------------------------------

print("\n4. DATA TYPES")
print("-" * 40)

print(df.dtypes)


# --------------------------------------------------
# 7. Check missing values
# --------------------------------------------------

print("\n5. MISSING VALUES")
print("-" * 40)

missing_values = df.isnull().sum()

missing_values = missing_values[missing_values > 0]

if missing_values.empty:
    print("No missing values found.")
else:
    print(missing_values.sort_values(ascending=False))


# --------------------------------------------------
# 8. Check duplicate records
# --------------------------------------------------

print("\n6. DUPLICATE RECORDS")
print("-" * 40)

duplicate_count = df.duplicated().sum()

print(f"Total duplicate rows: {duplicate_count}")


# --------------------------------------------------
# 9. Statistical summary
# --------------------------------------------------

print("\n7. STATISTICAL SUMMARY")
print("-" * 40)

print(df.describe(include="all").T)


print("\n8. TARGET COLUMN ANALYSIS")
print("-" * 40)

proxy = (df["age"] >= AGE_THRESHOLD).astype("int8")
print(f"Target: {TARGET_COLUMN} (synthetic; age >= {AGE_THRESHOLD})")
print(proxy.value_counts().sort_index().rename("count"))
print("This target is not an observed player-churn outcome.")


# --------------------------------------------------
# 11. Save exploration reports
# --------------------------------------------------

missing_report = df.isnull().sum().reset_index()

missing_report.columns = ["column", "missing_count"]

missing_report.to_csv(
    REPORT_PATH / "missing_values_report.csv",
    index=False
)

df.describe(include="all").T.to_csv(
    REPORT_PATH / "statistical_summary.csv"
)

print("\n9. REPORTS SAVED")
print("-" * 40)

print("reports/missing_values_report.csv")
print("reports/statistical_summary.csv")


# --------------------------------------------------
# 12. Completion message
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATASET EXPLORATION COMPLETED SUCCESSFULLY")
print("=" * 60)