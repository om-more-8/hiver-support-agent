"""
LLM-based intent classifier. Few-shot, not fine-tuned — no time budget
for training, and few-shot with a fixed taxonomy is the right tool here.

USAGE:
    from src.intents.classifier import IntentClassifier
    clf = IntentClassifier()
    clf.classify("my order still hasn't arrived")
"""
import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.llm_client import chat
from src.config import GOLDEN_SET_PATH

INTENT_TAXONOMY = [
    "delivery_delay", "product_or_shipping_inquiry", "billing_dispute",
    "positive_feedback", "account_login_issue", "refund_request",
    "customer_discontent", "damaged_or_defective_item",
    "prime_membership_question", "return_issue_due_to_courier",
]

SYSTEM_PROMPT = f"""Classify the customer support tweet into EXACTLY ONE of these intents:
{', '.join(INTENT_TAXONOMY)}

Respond with ONLY valid JSON, no other text: {{"intent": "<one of the above>", "confidence": <0.0-1.0>}}
"""


class IntentClassifier:
    def __init__(self):
        self.taxonomy = INTENT_TAXONOMY

    def classify(self, customer_text: str) -> dict:
        try:
            raw = chat(prompt=f"Tweet: {customer_text}", system=SYSTEM_PROMPT, temperature=0.0)
            raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
            result = json.loads(raw)
            intent = result.get("intent", "")
            confidence = float(result.get("confidence", 0.5))
            if intent not in self.taxonomy:
                return {"intent": "product_or_shipping_inquiry", "confidence": 0.3,
                         "note": f"LLM returned off-taxonomy intent '{intent}', fell back to default"}
            return {"intent": intent, "confidence": confidence}
        except Exception as e:
            return {"intent": "product_or_shipping_inquiry", "confidence": 0.0,
                     "note": f"classification error: {e}"}


if __name__ == "__main__":
    clf = IntentClassifier()
    tests = [
        "My order still hasn't arrived after 2 weeks, where is it??",
        "This is the third time I've contacted you about my refund, I want a lawyer involved",
        "Thanks so much for the quick help, really appreciate it!",
    ]
    for t in tests:
        print(f"\nInput: {t}")
        print(f"Result: {clf.classify(t)}")