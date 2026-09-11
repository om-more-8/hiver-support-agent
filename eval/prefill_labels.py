"""
Reads your IN-PROGRESS data/golden_set_for_labeling.csv and, for any
row still missing intent/should_escalate/escalation_reason, asks the
LLM for a SUGGESTED label. Already-labeled rows are left untouched.

Writes to a SEPARATE file (data/golden_set_draft.csv) — never
overwrites your real labeling file, and this draft never gets
committed to git (already in .gitignore).

YOU review every suggested row and copy corrected values into your
real golden_set_for_labeling.csv. The suggestion column itself never
ships anywhere.

USAGE:
    python eval/prefill_labels.py
"""
import pandas as pd
import json
import re
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import ROOT
from src.llm_client import chat

SYSTEM_PROMPT = """You label customer support tweets for an Amazon-style e-commerce brand.
Given a customer's tweet, respond with ONLY valid JSON, no other text:
{"intent": "<lowercase_with_underscores>", "should_escalate": true|false, "escalation_reason": "<short phrase>"}

Common intents seen in this data: delivery_delay, refund_request, damaged_or_defective_item,
billing_dispute, account_login_issue, product_or_shipping_inquiry, prime_membership_question,
positive_feedback. Use one of these if it fits; otherwise invent a similarly-formatted new one.

should_escalate = true for: repeated/unresolved contact, explicit anger or threats, safety issues,
ambiguous cases where a wrong auto-reply would make things worse.
should_escalate = false for: routine status checks, simple factual questions, standard first-time
complaints with an obvious resolution path.
"""


def suggest_label(customer_text: str) -> dict:
    try:
        raw = chat(prompt=f"Tweet: {customer_text}", system=SYSTEM_PROMPT, temperature=0.0)
        raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        return json.loads(raw)
    except Exception as e:
        return {"intent": "", "should_escalate": "", "escalation_reason": f"[LLM error: {e}]"}


def is_empty(val) -> bool:
    return pd.isna(val) or str(val).strip() == ""


def main():
    in_path = ROOT / "data" / "golden_set_for_labeling.csv"
    out_path = ROOT / "data" / "golden_set_draft.csv"

    df = pd.read_csv(in_path, encoding="utf-8-sig")
    df = df.astype(object)

    needs_label = df.apply(
        lambda row: is_empty(row.get("intent")) or is_empty(row.get("should_escalate"))
        or is_empty(row.get("escalation_reason")), axis=1
    )
    print(f"{needs_label.sum()} of {len(df)} rows still need labels — generating suggestions for those.")

    for idx in tqdm(df[needs_label].index, desc="Suggesting labels"):
        suggestion = suggest_label(df.loc[idx, "customer_text"])
        df.loc[idx, "intent"] = suggestion.get("intent", "")
        df.loc[idx, "should_escalate"] = str(suggestion.get("should_escalate", "")).upper()
        df.loc[idx, "escalation_reason"] = suggestion.get("escalation_reason", "")

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nWrote suggestions to {out_path}")
    print("This is a DRAFT — review every suggested row (rows that were previously empty),")
    print("correct anything wrong, then copy/save your final reviewed version as")
    print("data/golden_set_for_labeling.csv before running eval/csv_to_jsonl.py")
    print("\nThe draft file is gitignored — it will never be committed or visible to anyone.")


if __name__ == "__main__":
    main()