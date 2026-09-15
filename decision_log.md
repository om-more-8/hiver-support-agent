# Decision Log

Plain list of non-obvious decisions made during this project, and the reasoning behind each.

1. **Chose AmazonHelp as the brand** — highest outbound tweet volume among candidate brands in the dataset (154,976 reconstructed pairs before filtering), giving enough data for both retrieval grounding and a representative golden set sample.

2. **Filtered to English-only customer messages** — the raw AmazonHelp data was ~27% non-English (Japanese, Spanish, Hindi, German, and others). Kept the intent taxonomy and evaluation consistent within the time budget rather than building multilingual handling from the start. Documented as a scoped-out feature, not an oversight (see Report, Section 5).

3. **Golden set sampled via stratified random sampling by text-length bucket** (short/medium/long), not pure random — avoids skewing the evaluation set toward simple one-line messages, with a fixed random seed (42) for reproducibility.

4. **Consolidated near-duplicate intent categories during labeling** — `customer_discontent`, `customer_service_complaint`, `customer_experience_issue`, and `customer_service_issue` were all functionally the same (generic dissatisfaction, no specific resolvable issue) and were merged into a single `customer_discontent` category. Kept categories tied to actionable resolution paths, not just descriptive labels.

5. **`escalation_reason` made optional for `should_escalate=FALSE` rows**, required only for TRUE rows or genuinely borderline FALSE calls — avoids filler text ("no issue") that added no evaluation or reporting value, while still requiring justification for the calls that actually matter.

6. **Used AI-assisted labeling (LLM-suggested draft labels, human-reviewed) for the golden set, not pure LLM auto-labeling** — every suggested label was manually reviewed and corrected before being accepted. Avoided the circularity of evaluating an LLM-based system against ground truth that was also LLM-generated without human review.

7. **Few-shot LLM intent classification instead of fine-tuning** — no time budget for training/fine-tuning within the assignment window; few-shot prompting against a fixed, hand-defined taxonomy was the higher-leverage choice.

8. **RAG-style grounded reply generation instead of free LLM generation** — replies are generated using the 2-3 most similar historical (customer message → brand reply) pairs as in-context examples, so responses are grounded in how the brand actually resolved similar issues rather than the LLM's general knowledge.

9. **Escalation logic was rewritten mid-project after baseline comparison exposed a real bug** — the original rule-based escalation logic (keyword matching, mirroring the simple baseline) scored identically to the simple baseline and had recall of only 0.26, missing 74% of true escalations. Replaced with LLM judgment plus a small keyword safety net for unambiguous red-flag terms (lawyer, chargeback, sue). Recall improved to 0.95. This is direct evidence that requiring baseline comparison caught a real defect that inspecting the AI system alone would have missed.

10. **Escalation recall was prioritized over precision** — for a public-facing brand account, a missed escalation (a wrong auto-reply to an urgent/angry public complaint) was judged a worse failure than an unnecessary escalation (wasted human agent time on a routine message). This tradeoff is explicit, not incidental — see Report, Section 4.

11. **Embeddings cache (`sentence-transformers/all-MiniLM-L6-v2` output over ~112k pairs) is excluded from git** — the file is 164MB, over GitHub's 100MB limit, and is fully regeneratable from the raw dataset on first run (~5-7 minutes). Documented in the README rather than committed.

12. **No FastAPI/React deployment layer was built** — the assignment's deliverable is a repo + report reviewed live, not a deployed product. Kept scope to a runnable CLI pipeline (`src/pipeline.py`) to protect the <15-minute reproduction requirement, rather than adding deployment complexity with no required payoff.

13. **Banking77 (the optional secondary dataset) was not used** — its 77 banking-specific intents (loans, card issues, transfers) don't transfer to e-commerce/shipping support and would have diluted rather than strengthened the taxonomy for this brand.

14. **Off-taxonomy LLM classification outputs are caught and defaulted, not allowed to silently corrupt results** — if the intent classifier LLM returns a category outside the fixed taxonomy (rare, but possible), the pipeline falls back to a default category rather than letting an invalid label propagate into evaluation.

15. **gpt-4o-mini via OpenRouter was used for all LLM calls** (classification, reply generation, escalation, judging) — cost-efficient for the call volume required (roughly 600+ calls across golden-set evaluation and judging), with quality sufficient for few-shot classification and grounded generation at this scale.