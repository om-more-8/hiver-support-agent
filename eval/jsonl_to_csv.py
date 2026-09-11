"""
Converts data/golden_set_unlabeled.jsonl into a CSV you can open and
label in Excel or Google Sheets.

USAGE:
    python eval/jsonl_to_csv.py
Writes: data/golden_set_for_labeling.csv
"""
import json
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import ROOT


def main():
    in_path = ROOT / "data" / "golden_set_unlabeled.jsonl"
    out_path = ROOT / "data" / "golden_set_for_labeling.csv"

    records = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)

    # Reorder so labeling columns are right next to what you're reading —
    # minimizes eye travel while going through rows.
    cols = ["customer_tweet_id", "customer_text", "intent",
            "should_escalate", "escalation_reason", "brand_reply_text_actual"]
    df = df[cols]

    df.to_csv(out_path, index=False, encoding="utf-8-sig")  # utf-8-sig so Excel shows non-ASCII correctly
    print(f"Wrote {len(df)} rows to {out_path}")
    print("\nHOW TO LABEL:")
    print("1. Open this CSV in Excel or Google Sheets.")
    print("2. For each row, read 'customer_text' and fill in:")
    print("   - intent: short lowercase_with_underscores label, e.g. delivery_delay, refund_request, account_login")
    print("     (define your own categories as you go — after ~20 rows you'll see natural clusters emerge)")
    print("   - should_escalate: TRUE or FALSE")
    print("   - escalation_reason: one short phrase, e.g. 'angry tone + repeated contact' or 'routine status check'")
    print("3. Ignore 'brand_reply_text_actual' — it's just reference context (what really happened), not something to edit.")
    print("4. Save as CSV when done (keep the same filename).")
    print("5. Run: python eval/csv_to_jsonl.py   to convert it back + validate it.")


if __name__ == "__main__":
    main()