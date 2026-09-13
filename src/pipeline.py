"""
THE AI SUPPORT AGENT. Ties together:
  1. Intent classification (LLM, few-shot)
  2. Retrieval (embeddings over historical resolved threads)
  3. Reply generation (RAG, grounded in retrieved examples)
  4. Escalation decision (rule-based on intent + confidence + risk signals)

This is what gets scored against the golden set in eval/metrics.py,
and what you'll run live in the interview.

USAGE (CLI):
    python src/pipeline.py --tweet "my order still hasn't arrived"
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.intents.classifier import IntentClassifier
from src.retrieval.embed_and_search import Retriever
from src.reply.generate_reply import ReplyGenerator
from src.escalation.decide import EscalationDecider


class SupportAgent:
    def __init__(self):
        print("Loading support agent (embeddings model + retrieval index)...")
        self.classifier = IntentClassifier()
        self.retriever = Retriever()
        self.reply_gen = ReplyGenerator(retriever=self.retriever)
        self.escalation_decider = EscalationDecider()
        print("Agent ready.\n")

    def predict(self, customer_text: str) -> dict:
        intent_result = self.classifier.classify(customer_text)
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]

        reply_result = self.reply_gen.generate(customer_text, intent=intent)

        escalation_result = self.escalation_decider.decide(customer_text, intent, confidence)

        return {
            "customer_text": customer_text,
            "intent": intent,
            "intent_confidence": confidence,
            "reply": reply_result["reply"],
            "grounded_on": reply_result["grounded_on"],
            "escalate": escalation_result["escalate"],
            "escalation_reason": escalation_result["escalation_reason"],
        }


def main():
    parser = argparse.ArgumentParser(description="Run the AI support agent on a single tweet.")
    parser.add_argument("--tweet", type=str, required=True, help="Customer message to process")
    args = parser.parse_args()

    agent = SupportAgent()
    result = agent.predict(args.tweet)

    print("=" * 60)
    print(f"CUSTOMER: {result['customer_text']}")
    print("=" * 60)
    print(f"Intent:            {result['intent']} (confidence: {result['intent_confidence']:.2f})")
    print(f"Escalate:          {result['escalate']}")
    print(f"Escalation reason: {result['escalation_reason']}")
    print(f"\nGenerated reply:\n  {result['reply']}")
    print(f"\nGrounded on {len(result['grounded_on'])} similar historical cases")


if __name__ == "__main__":
    main()