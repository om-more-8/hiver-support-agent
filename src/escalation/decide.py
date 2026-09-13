"""
Decides auto-handle vs escalate-to-human, with a stated reason.
Rule-based on top of intent + classifier confidence + risk keyword
signals — deliberately NOT another LLM call. An escalation gate should
be predictable and auditable, not another black box; this mirrors the
keyword-boosted risk scoring approach from CogniClause.

USAGE:
    from src.escalation.decide import EscalationDecider
    d = EscalationDecider()
    d.decide(text, intent="delivery_delay", confidence=0.9)
"""
import re

RISK_KEYWORDS = [
    "lawyer", "chargeback", "sue", "legal action", "third time", "again and again",
    "still not resolved", "no response", "speak to a manager", "unacceptable",
    "doesn't care", "arrogant", "disgusted", "worst", "never coming back",
]

HIGH_STAKES_INTENTS = {"billing_dispute", "customer_discontent"}
LOW_CONFIDENCE_THRESHOLD = 0.5


class EscalationDecider:
    def decide(self, customer_text: str, intent: str, confidence: float) -> dict:
        text_lower = customer_text.lower()
        reasons = []

        risk_hits = [kw for kw in RISK_KEYWORDS if kw in text_lower]
        if risk_hits:
            reasons.append(f"risk keyword(s) detected: {risk_hits}")

        if re.search(r"[A-Z]{4,}", customer_text):
            reasons.append("shouting/all-caps detected")

        if confidence < LOW_CONFIDENCE_THRESHOLD:
            reasons.append(f"low classifier confidence ({confidence:.2f})")

        if intent in HIGH_STAKES_INTENTS and confidence < 0.7:
            reasons.append(f"high-stakes intent '{intent}' with moderate confidence")

        should_escalate = len(reasons) > 0
        final_reason = "; ".join(reasons) if reasons else "routine message, no risk signals, confident classification"

        return {"escalate": should_escalate, "escalation_reason": final_reason}


if __name__ == "__main__":
    d = EscalationDecider()
    tests = [
        ("my order is a bit late", "delivery_delay", 0.9),
        ("THIS IS THE THIRD TIME, I want a lawyer involved NOW", "customer_discontent", 0.85),
        ("does this include 2 day shipping?", "product_or_shipping_inquiry", 0.95),
    ]
    for text, intent, conf in tests:
        print(f"\nInput: {text}")
        print(f"Decision: {d.decide(text, intent, conf)}")