"""
Decides auto-handle vs escalate-to-human, with a stated reason.

REVISED APPROACH (see decision log): the original pure rule-based
version scored identically to the simple baseline (same keyword logic)
and had weak recall (0.26) — missing most cases that should actually
escalate. This version uses LLM judgment for the nuanced call, with a
small keyword safety net that FORCES escalate=True on unambiguous
red-flag terms regardless of what the LLM says — a deliberate design
choice so guaranteed-risk language never slips through on an LLM
misjudgment.

USAGE:
    from src.escalation.decide import EscalationDecider
    d = EscalationDecider()
    d.decide(text, intent="delivery_delay", confidence=0.9)
"""
import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.llm_client import chat

FORCE_ESCALATE_KEYWORDS = [
    "lawyer", "chargeback", "sue", "legal action", "attorney",
]

SYSTEM_PROMPT = """You decide whether an incoming customer support tweet should be
auto-handled by a bot or escalated to a human agent.

Escalate to a human when: the issue is unresolved after repeated contact, the customer
shows real anger or frustration (not just a routine complaint), there's ambiguity a wrong
auto-reply could make worse, or the message involves billing/legal/safety risk.

Auto-handle when: it's a routine question, a first-time standard complaint with an obvious
resolution path, or positive/neutral feedback needing no real action.

Respond with ONLY valid JSON, no other text:
{"escalate": true|false, "reason": "<short specific phrase, not generic>"}
"""


class EscalationDecider:
    def decide(self, customer_text: str, intent: str, confidence: float) -> dict:
        text_lower = customer_text.lower()
        forced_hits = [kw for kw in FORCE_ESCALATE_KEYWORDS if kw in text_lower]

        try:
            prompt = f"Customer tweet (classified intent: {intent}): {customer_text}"
            raw = chat(prompt=prompt, system=SYSTEM_PROMPT, temperature=0.0)
            raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
            result = json.loads(raw)
            escalate = bool(result.get("escalate", False))
            reason = str(result.get("reason", "")).strip()
        except Exception as e:
            escalate, reason = False, f"[LLM error, defaulted to no escalation: {e}]"

        if forced_hits and not escalate:
            escalate = True
            reason = f"safety-net override — red-flag term(s) detected: {forced_hits}"
        elif forced_hits:
            reason = f"{reason} (also matched red-flag term(s): {forced_hits})"

        return {"escalate": escalate, "escalation_reason": reason}


if __name__ == "__main__":
    d = EscalationDecider()
    tests = [
        ("my order is a bit late", "delivery_delay", 0.9),
        ("THIS IS THE THIRD TIME, I want a lawyer involved NOW", "customer_discontent", 0.85),
        ("does this include 2 day shipping?", "product_or_shipping_inquiry", 0.95),
        ("still no response, this is ridiculous, third time contacting you", "customer_discontent", 0.8),
    ]
    for text, intent, conf in tests:
        print(f"\nInput: {text}")
        print(f"Decision: {d.decide(text, intent, conf)}")