"""
Surfaces exactly what needs review before moving to the real pipeline:
1. Rare intent categories (support <= 2) — likely near-duplicates that
   should be merged into your main taxonomy.
2. Every should_escalate=TRUE row — quick skim to sanity-check the
   escalation rate isn't inflated by over-cautious labeling.

USAGE:
    python eval/audit_golden_set.py
"""
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import GOLDEN_SET_PATH

RARE_THRESHOLD = 2  # intents with this many examples or fewer get flagged


def main():
    records = [json.loads(l) for l in open(GOLDEN_SET_PATH, encoding="utf-8")]
    n = len(records)

    intent_counts = Counter(r["intent"] for r in records)
    print(f"{'='*60}\nINTENT DISTRIBUTION ({len(intent_counts)} categories, {n} examples)\n{'='*60}")
    for intent, count in intent_counts.most_common():
        flag = "  <-- RARE, likely merge target" if count <= RARE_THRESHOLD else ""
        print(f"  {count:4d}  {intent}{flag}")

    rare_intents = {i for i, c in intent_counts.items() if c <= RARE_THRESHOLD}
    if rare_intents:
        print(f"\n{'='*60}\nROWS WITH RARE INTENTS — decide which main category each belongs to\n{'='*60}")
        for r in records:
            if r["intent"] in rare_intents:
                print(f"\n[{r['intent']}] (tweet_id={r['customer_tweet_id']})")
                print(f"  \"{r['customer_text'][:150]}\"")

    escalate_true = [r for r in records if r["should_escalate"]]
    rate = len(escalate_true) / n
    print(f"\n{'='*60}\nESCALATION RATE: {len(escalate_true)}/{n} = {rate:.1%} marked TRUE\n{'='*60}")
    if rate > 0.4:
        print("NOTE: this is high for typical support volume — worth skimming for over-cautious labels.\n")

    print("All should_escalate=TRUE rows (skim for ones that look borderline/routine, not truly urgent):\n")
    for r in escalate_true:
        print(f"[{r['intent']}] (id={r['customer_tweet_id']}) reason: {r['escalation_reason']}")
        print(f"  \"{r['customer_text'][:150]}\"\n")


if __name__ == "__main__":
    main()