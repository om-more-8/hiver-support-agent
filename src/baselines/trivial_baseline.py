"""
TRIVIAL BASELINE — the floor. Near-zero intelligence.
- intent: always the single most common intent (majority class)
- reply: one generic canned message for every tweet
- escalate: always the majority answer in the golden set

If the real system doesn't clearly beat this, something is wrong.
"""
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import GOLDEN_SET_PATH

CANNED_REPLY = "Thanks for reaching out! We're looking into this and will follow up with you shortly."


class TrivialBaseline:
    def __init__(self, golden_set_path=GOLDEN_SET_PATH):
        records = [json.loads(l) for l in open(golden_set_path, encoding="utf-8")]
        self.majority_intent = Counter(r["intent"] for r in records).most_common(1)[0][0]
        self.majority_escalate = Counter(r["should_escalate"] for r in records).most_common(1)[0][0]

    def predict(self, customer_text: str) -> dict:
        return {
            "intent": self.majority_intent,
            "reply": CANNED_REPLY,
            "escalate": self.majority_escalate,
            "escalation_reason": "trivial baseline — no reasoning, fixed default",
        }


if __name__ == "__main__":
    baseline = TrivialBaseline()
    print(f"Majority intent: {baseline.majority_intent}")
    print(f"Majority escalate: {baseline.majority_escalate}")
    print(f"\nSample prediction: {baseline.predict('any tweet text at all')}")