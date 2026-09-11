"""
Samples N examples from the processed pairs for you to hand-label.

Sampling method (documented for your report):
- Random sample, but stratified by TEXT LENGTH BUCKET (short/medium/long)
  as a rough proxy for complaint complexity, so the golden set isn't
  skewed toward simple one-line messages.
- Fixed random seed for reproducibility.

USAGE:
    python eval/build_golden_set.py
Writes: data/golden_set_unlabeled.jsonl  (you then label it -> golden_set.jsonl)
"""
import pandas as pd
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import PROCESSED_DIR, BRAND, GOLDEN_SET_SIZE, RANDOM_SEED, ROOT


def length_bucket(text: str) -> str:
    n = len(str(text))
    if n < 60:
        return "short"
    elif n < 150:
        return "medium"
    else:
        return "long"


def main():
    pairs_path = PROCESSED_DIR / f"{BRAND}_pairs.csv"
    df = pd.read_csv(pairs_path)
    print(f"Loaded {len(df)} pairs for brand {BRAND}")

    df["length_bucket"] = df["customer_text"].apply(length_bucket)
    print("Length bucket distribution:")
    print(df["length_bucket"].value_counts())

    # Stratified sample: proportionally sample from each bucket
    n_target = min(GOLDEN_SET_SIZE, len(df))
    sampled = df.groupby("length_bucket", group_keys=False).apply(
        lambda g: g.sample(
            n=min(len(g), max(1, round(n_target * len(g) / len(df)))),
            random_state=RANDOM_SEED,
        )
    )

    # Top up/trim to exact target if rounding drifted
    if len(sampled) > n_target:
        sampled = sampled.sample(n=n_target, random_state=RANDOM_SEED)
    elif len(sampled) < n_target:
        remaining = df.drop(sampled.index)
        extra = remaining.sample(n=n_target - len(sampled), random_state=RANDOM_SEED)
        sampled = pd.concat([sampled, extra])

    sampled = sampled.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)  # shuffle

    out_path = ROOT / "data" / "golden_set_unlabeled.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in sampled.iterrows():
            record = {
                "customer_tweet_id": row["customer_tweet_id"],
                "customer_text": row["customer_text"],
                "brand_reply_text_actual": row["brand_reply_text"],  # what the brand really said — useful reference, not the "correct" label
                # --- YOU FILL THESE IN ---
                "intent": "",             # e.g. "delivery_delay", "refund_request" — define your own taxonomy from the data
                "should_escalate": None,  # true / false
                "escalation_reason": "",  # short reason
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(sampled)} examples to {out_path}")
    print("Next: open this file and fill in 'intent', 'should_escalate', 'escalation_reason' for each row.")
    print("Once done, save it as data/golden_set.jsonl")


if __name__ == "__main__":
    main()