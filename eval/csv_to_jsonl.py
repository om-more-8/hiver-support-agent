"""
Converts your labeled data/golden_set_for_labeling.csv back into
data/golden_set.jsonl — the file every eval script downstream reads.

Validates as it goes so mistakes get caught NOW, not silently baked
into your eval numbers later.

USAGE:
    python eval/csv_to_jsonl.py
Writes: data/golden_set.jsonl
"""
import pandas as pd
import json
import re
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import ROOT

VALID_ESCALATE_VALUES = {"true": True, "false": False, "1": True, "0": False}


def main():
    in_path = ROOT / "data" / "golden_set_for_labeling.csv"
    out_path = ROOT / "data" / "golden_set.jsonl"

    df = pd.read_csv(in_path, encoding="utf-8-sig")
    errors = []

    for idx, row in df.iterrows():
        rownum = idx + 2  # +2 = header row + 1-indexing, matches what you'd see in Excel

        intent = str(row.get("intent", "")).strip()
        if not intent or intent.lower() == "nan":
            errors.append(f"Row {rownum}: missing 'intent'")
        elif not re.match(r"^[a-z0-9_]+$", intent):
            errors.append(f"Row {rownum}: 'intent' should be lowercase_with_underscores, got '{intent}'")

        escalate_raw = str(row.get("should_escalate", "")).strip().lower()
        if escalate_raw not in VALID_ESCALATE_VALUES:
            errors.append(f"Row {rownum}: 'should_escalate' must be TRUE or FALSE, got '{row.get('should_escalate')}'")

        reason = str(row.get("escalation_reason", "")).strip()
        is_escalate_true = VALID_ESCALATE_VALUES.get(escalate_raw, False)
        if is_escalate_true and (not reason or reason.lower() == "nan"):
            errors.append(f"Row {rownum}: 'escalation_reason' required when should_escalate=TRUE")

    if errors:
        print(f"FOUND {len(errors)} LABELING ISSUES — fix these in the CSV and rerun:\n")
        for e in errors[:30]:
            print(f"  - {e}")
        if len(errors) > 30:
            print(f"  ... and {len(errors) - 30} more")
        print("\nNo output written. Fix the CSV and rerun this script.")
        return

    # Also flag (not block) intent categories that appear only once —
    # possibly a typo/inconsistent label, worth a manual glance.
    intent_counts = df["intent"].value_counts()
    rare = intent_counts[intent_counts == 1]
    if len(rare) > 0:
        print(f"NOTE: {len(rare)} intent labels appear only once — double check these aren't typos "
              f"of a more common category: {list(rare.index)}\n")

    print(f"All {len(df)} rows passed validation. Intent categories found:")
    print(df["intent"].value_counts().to_string())

    records = []
    for _, row in df.iterrows():
        reason_val = str(row["escalation_reason"]).strip()
        if reason_val.lower() == "nan":
            reason_val = ""
        records.append({
            "customer_tweet_id": row["customer_tweet_id"],
            "customer_text": row["customer_text"],
            "intent": str(row["intent"]).strip(),
            "should_escalate": VALID_ESCALATE_VALUES[str(row["should_escalate"]).strip().lower()],
            "escalation_reason": reason_val,
            "brand_reply_text_actual": row["brand_reply_text_actual"],
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} validated examples to {out_path}")
    print("Golden set is ready. Next: baselines + intent classifier.")


if __name__ == "__main__":
    main()