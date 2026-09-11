"""
SIMPLE BASELINE — a genuine non-LLM attempt. What a decent engineer
ships in a day without any AI/LLM involved.
- intent: keyword/rule matching
- reply: most textually similar past resolved tweet (TF-IDF), reuse its real reply
- escalate: rule-based on risk keywords/signals

Your real AI system needs to clearly beat THIS, not just the trivial one —
otherwise the LLM/embedding complexity isn't earning its keep.
"""
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import PROCESSED_DIR, BRAND

# Keyword rules — edit these to match the intent taxonomy you actually used
# when labeling your golden set.
INTENT_KEYWORDS = {
    "refund_request": ["refund", "money back", "reimburse"],
    "delivery_delay": ["still hasn't arrived", "where is my", "delayed", "late delivery", "hasn't come"],
    "damaged_or_defective_item": ["broken", "damaged", "defective", "arrived smashed", "doesn't work"],
    "billing_dispute": ["charged twice", "double charge", "overcharged", "wrong amount"],
    "account_login_issue": ["can't log in", "cannot login", "password reset", "invalid password"],
    "prime_membership_question": ["prime member", "prime membership", "prime benefits"],
    "positive_feedback": ["thank you", "thanks so much", "love", "great service", "appreciate"],
}
DEFAULT_INTENT = "product_or_shipping_inquiry"

ESCALATION_KEYWORDS = [
    "lawyer", "chargeback", "sue", "legal", "third time", "again and again",
    "still not resolved", "no response", "manager", "unacceptable", "worst",
]


def keyword_intent(text: str) -> str:
    text_lower = text.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return intent
    return DEFAULT_INTENT


def rule_based_escalate(text: str) -> tuple[bool, str]:
    text_lower = text.lower()
    hits = [kw for kw in ESCALATION_KEYWORDS if kw in text_lower]
    has_caps_shout = bool(re.search(r"[A-Z]{4,}", text))
    if hits:
        return True, f"risk keyword(s) matched: {hits}"
    if has_caps_shout:
        return True, "shouting/all-caps detected"
    return False, "no risk signals detected"


class SimpleBaseline:
    def __init__(self, brand: str = BRAND):
        pairs_path = PROCESSED_DIR / f"{brand}_pairs.csv"
        self.pairs = pd.read_csv(pairs_path)
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.pairs["customer_text"].astype(str))

    def most_similar_reply(self, customer_text: str) -> str:
        query_vec = self.vectorizer.transform([customer_text])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        best_idx = sims.argmax()
        return self.pairs.iloc[best_idx]["brand_reply_text"]

    def predict(self, customer_text: str) -> dict:
        intent = keyword_intent(customer_text)
        escalate, reason = rule_based_escalate(customer_text)
        reply = self.most_similar_reply(customer_text)
        return {
            "intent": intent,
            "reply": reply,
            "escalate": escalate,
            "escalation_reason": reason,
        }


if __name__ == "__main__":
    baseline = SimpleBaseline()
    test_texts = [
        "I need a refund for my order, it's been 2 weeks",
        "This is the THIRD time I've contacted you, I want a lawyer involved",
        "Thanks so much for the quick help!",
    ]
    for t in test_texts:
        print(f"\nInput: {t}")
        print(f"Prediction: {baseline.predict(t)}")