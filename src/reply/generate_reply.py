"""
Generates a reply grounded in how the brand ACTUALLY resolved similar
past issues (via retrieval), rather than free-generating from the LLM's
own knowledge. This is the "grounded in historical resolutions" part
of the assignment brief.

USAGE:
    from src.reply.generate_reply import ReplyGenerator
    gen = ReplyGenerator()
    gen.generate("my order hasn't arrived", intent="delivery_delay")
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.llm_client import chat
from src.retrieval.embed_and_search import Retriever

SYSTEM_PROMPT = """You draft customer support replies for an e-commerce brand's Twitter support account.
You will be given a new customer message and 2-3 examples of how this brand has
ACTUALLY resolved similar issues in the past. Write a reply that:
- Matches the brand's real tone and typical resolution pattern from the examples
- Is specific to the new customer's actual message (don't just copy an example verbatim)
- Is concise, in the style of a real support tweet (1-3 sentences)

Respond with ONLY the reply text, nothing else — no preamble, no quotes.
"""


class ReplyGenerator:
    def __init__(self, retriever: Retriever = None):
        self.retriever = retriever or Retriever()

    def generate(self, customer_text: str, intent: str = "") -> dict:
        similar = self.retriever.search(customer_text, k=3)

        examples_block = "\n\n".join(
            f"Example {i+1}:\nCustomer: {ex['customer_text']}\nBrand's actual reply: {ex['brand_reply_text']}"
            for i, ex in enumerate(similar)
        )

        prompt = f"""New customer message (intent: {intent}):
{customer_text}

Historical examples of how this brand resolved similar issues:
{examples_block}

Write the reply to the new customer message now."""

        reply_text = chat(prompt=prompt, system=SYSTEM_PROMPT, temperature=0.3)

        return {
            "reply": reply_text,
            "grounded_on": [ex["customer_text"] for ex in similar],
            "retrieval_similarity_scores": [round(ex["similarity"], 3) for ex in similar],
        }


if __name__ == "__main__":
    gen = ReplyGenerator()
    result = gen.generate("my order still hasn't arrived after 2 weeks", intent="delivery_delay")
    print(f"Generated reply: {result['reply']}")
    print(f"\nGrounded on {len(result['grounded_on'])} similar past cases "
          f"(similarity: {result['retrieval_similarity_scores']})")