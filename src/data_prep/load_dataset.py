"""
Loads the raw Twitter customer-support CSV, filters to one brand, and
reconstructs (customer_message -> brand_reply) pairs using the
response_tweet_id / in_reply_to_status_id link columns.

USAGE:
    python src/data_prep/load_dataset.py

Writes: data/processed/{BRAND}_pairs.csv
Columns: customer_text, brand_reply_text, customer_tweet_id, brand_tweet_id
"""
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import RAW_CSV, PROCESSED_DIR, BRAND


def load_raw(path=RAW_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Either run `python src/data_prep/make_mock_data.py` "
            f"for a test file, or download the real dataset from Kaggle "
            f"(thoughtvector/customer-support-on-twitter) and save it here as twcs.csv"
        )
    df = pd.read_csv(path, dtype={"tweet_id": str, "response_tweet_id": str,
                                   "in_reply_to_status_id": str})
    return df


def reconstruct_pairs(df: pd.DataFrame, brand: str = BRAND) -> pd.DataFrame:
    """
    Real dataset logic: a brand-side tweet has inbound == False and
    author_id == brand. Its in_reply_to_status_id points to the customer
    tweet_id it's replying to. We join on that.
    """
    df["inbound"] = df["inbound"].astype(str).str.lower().isin(["true", "1"])

    brand_replies = df[(df["inbound"] == False) & (df["author_id"] == brand)].copy()
    customer_msgs = df[df["inbound"] == True].copy()

    merged = brand_replies.merge(
        customer_msgs,
        left_on="in_reply_to_status_id",
        right_on="tweet_id",
        suffixes=("_brand", "_customer"),
    )

    pairs = merged[[
        "tweet_id_customer", "text_customer", "tweet_id_brand", "text_brand"
    ]].rename(columns={
        "tweet_id_customer": "customer_tweet_id",
        "text_customer": "customer_text",
        "tweet_id_brand": "brand_tweet_id",
        "text_brand": "brand_reply_text",
    })

    pairs = pairs.dropna(subset=["customer_text", "brand_reply_text"])
    pairs = pairs.drop_duplicates(subset=["customer_tweet_id"])
    return pairs.reset_index(drop=True)


def main():
    print(f"Loading raw data from {RAW_CSV}...")
    df = load_raw()
    print(f"Loaded {len(df)} total rows.")

    print(f"Filtering + reconstructing pairs for brand: {BRAND}")
    pairs = reconstruct_pairs(df, BRAND)
    print(f"Reconstructed {len(pairs)} customer->brand reply pairs.")

    if len(pairs) == 0:
        print(f"WARNING: 0 pairs found for brand '{BRAND}'. Check that this "
              f"author_id exists in the dataset (case-sensitive) — inspect "
              f"df['author_id'].value_counts() in notebooks/eda.ipynb.")

    out_path = PROCESSED_DIR / f"{BRAND}_pairs.csv"
    pairs.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
