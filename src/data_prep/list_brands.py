"""
Run this once against the REAL dataset to see the exact brand names
(author_id values) available, ranked by tweet volume. Copy the exact
string (case-sensitive) into your .env as BRAND=<value> or into
src/config.py's default.

USAGE:
    python src/data_prep/list_brands.py
"""
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import RAW_CSV

def main():
    df = pd.read_csv(RAW_CSV, usecols=["author_id", "inbound"],
                      dtype={"author_id": str})
    df["inbound"] = df["inbound"].astype(str).str.lower().isin(["true", "1"])
    brand_side = df[df["inbound"] == False]
    counts = brand_side["author_id"].value_counts().head(25)
    print("Top 25 brand author_ids by outbound tweet volume:\n")
    print(counts.to_string())
    print("\nPick one with high volume, copy the EXACT string into BRAND in src/config.py or .env")

if __name__ == "__main__":
    main()