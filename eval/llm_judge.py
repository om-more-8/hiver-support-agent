"""
LLM-as-judge for reply quality. Scores a generated reply 1-5 on a
fixed rubric, with a short rationale — not just a bare number, so you
can actually explain disagreements later.

USAGE:
    from eval.llm_judge import judge_reply
    judge_reply(customer_text, generated_reply)
"""
import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.llm_client import chat

SYSTEM_PROMPT = """You are evaluating a customer support reply for quality. Score it 1-5:

5 = Directly addresses the issue, appropriate tone, actionable/specific, doesn't overpromise
4 = Good reply, minor issues (slightly generic, or misses one small detail)
3 = Adequate but noticeably generic, or misses part of the actual issue
2 = Poor fit — addresses wrong aspect of the issue, or inappropriate tone
1 = Fails completely — irrelevant, or makes false promises/claims

Respond with ONLY valid JSON, no other text:
{"score": <1-5 integer>, "rationale": "<one short sentence>"}
"""


def judge_reply(customer_text: str, generated_reply: str) -> dict:
    prompt = f"Customer message: {customer_text}\n\nGenerated reply: {generated_reply}"
    try:
        raw = chat(prompt=prompt, system=SYSTEM_PROMPT, temperature=0.0)
        raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        result = json.loads(raw)
        return {"score": int(result.get("score", 3)), "rationale": str(result.get("rationale", ""))}
    except Exception as e:
        return {"score": None, "rationale": f"[judge error: {e}]"}


if __name__ == "__main__":
    tests = [
        ("my order still hasn't arrived after 2 weeks",
         "Sorry for the wait! Can you share your order number so we can check the tracking status?"),
        ("my order still hasn't arrived after 2 weeks",
         "Thanks for reaching out! We're looking into this and will follow up shortly."),
    ]
    for text, reply in tests:
        print(f"\nCustomer: {text}")
        print(f"Reply: {reply}")
        print(f"Judge: {judge_reply(text, reply)}")