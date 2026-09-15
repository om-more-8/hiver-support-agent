# Hiver SDE Intern Take-Home — Report

**Brand:** AmazonHelp (Twitter customer support)
**Dataset:** Kaggle "Customer Support on Twitter" (thoughtvector/customer-support-on-twitter), filtered to AmazonHelp, English-only (~112k customer-reply pairs after filtering)
**Golden evaluation set:** 200 hand-labelled examples (this author's own labels)

---

## 1. Problem Framing

**What "good" means for this brand:** AmazonHelp handles extremely high public support volume on Twitter, where replies are visible to everyone, not just the customer. For this brand, "good" means three things, in priority order:
1. **Never silently mishandle something that should go to a human** — a wrong auto-reply to an angry or unresolved public complaint is a reputation risk, not just a bad customer experience.
2. **Correctly identify what the customer actually needs** — a reply that's polite but addresses the wrong issue is nearly as bad as no reply.
3. **Sound like the brand actually sounds** — replies should be grounded in how AmazonHelp really resolves issues, not generic AI boilerplate.

Given this, escalation recall was treated as more important than escalation precision throughout this project (see Decision Log) — a missed escalation is a worse failure than an unnecessary one.

**What I chose not to build:**
- **Multilingual support.** The raw dataset was ~27% non-English (Japanese, Spanish, Hindi, German, etc.). I filtered to English-only to keep the intent taxonomy and evaluation consistent within the time budget. This is a real limitation, not an oversight — see Next Steps.
- **Fine-tuned models.** Intent classification and reply generation both use few-shot LLM prompting, not fine-tuning. Given the time constraints, few-shot with a fixed taxonomy was the higher-leverage choice.
- **A second secondary dataset (Banking77).** Explicitly marked optional in the brief; its banking-specific intents don't transfer to e-commerce support and would have diluted the taxonomy rather than strengthened it.
- **Multi-turn conversation handling.** Each prediction is made on a single customer message in isolation, not a full back-and-forth thread. Real support often spans multiple exchanges; this is a scoped-down version of the real problem.

---

## 2. Results vs. Baselines

Three systems were scored against the same 200-row golden set:

| System | Intent Accuracy | Intent Macro F1 | Escalation Precision | Escalation Recall | Escalation F1 |
|---|---|---|---|---|---|
| **Trivial baseline** (majority class, canned reply) | 29.0% | 0.045 | 0.595 | 1.000 | 0.746 |
| **Simple baseline** (keyword rules, TF-IDF retrieval, rule-based escalation) | 37.5% | 0.325 | 0.756 | 0.261 | 0.388 |
| **AI agent** (LLM intent classification + embedding retrieval + RAG reply generation + LLM escalation judgment) | **66.5%** | **0.628** | 0.837 | **0.950** | **0.890** |

**Reply quality** (LLM-as-judge, 1-5 scale, n=200): average score **3.62/5**.
**Judge-vs-human agreement** (n=40 human-scored sample): 70% exact match, **100% within-1-point**, Spearman correlation **0.689** (p<0.0001), mean absolute difference 0.30 points. See `results/judge_agreement_summary.json`.

**Headline finding:** the AI agent roughly doubles intent accuracy over both baselines and wins clearly on escalation F1. But this required one real mid-project fix — see below.

**A note on baseline design:** the trivial baseline's escalation F1 (0.746) briefly *beat* an earlier version of the AI agent, because that early version's escalation logic was accidentally almost identical to the simple baseline's keyword rules (recall was only 0.26 — missing 74% of true escalations). This was caught by comparing against baselines, not by inspecting the AI system in isolation, which is the entire reason baselines were required in the first place. The escalation logic was rewritten to use LLM judgment with a small keyword safety net (see Decision Log), raising recall to 0.95.

---

## 3. Failure Analysis — Top 5 Failure Modes

### 1. The catch-all intent category over-triggers on negative tone, regardless of actual issue
`customer_discontent` has precision of only 0.19 despite perfect recall (1.0) — the classifier routes almost any negatively-toned message here, even when a more specific category (`billing_dispute`, `delivery_delay`, `product_or_shipping_inquiry`) actually fits better. Real confusion counts: 8 true `delivery_delay`, 8 true `billing_dispute`, 8 true `product_or_shipping_inquiry`, and 3 true `positive_feedback` messages were all misclassified into `customer_discontent`.
**Hypothesis:** the LLM appears to weight emotional tone over topical content when both signals are present — an angry billing complaint gets classified by its anger, not its billing content.

### 2. The largest category (`product_or_shipping_inquiry`) is under-predicted, bleeding into `delivery_delay`
14 of the 51 true `product_or_shipping_inquiry` examples were misclassified as `delivery_delay` — the single largest confusion pair in the whole matrix.
**Hypothesis:** many genuine shipping *questions* ("does this include 2-day shipping?") share surface vocabulary with delivery *complaints* ("still hasn't arrived"), and the classifier appears to pattern-match on shipping-related keywords rather than distinguishing a question from a complaint.

### 3. Escalation over-triggers on ambiguous cases (precision 0.837, 22 false positives)
After rewriting the escalation logic to fix a recall problem (see above), the system now escalates 22 cases that didn't need it, out of 200. This is a deliberate tradeoff, not an accident — but it has a real cost: every unnecessary escalation is human agent time spent on something a bot could have handled.
**Hypothesis:** the LLM escalation judgment is calibrated toward caution given the prompt's framing, which is defensible for a public-facing brand but should be tunable per business risk tolerance in a real deployment.

### 4. Reply generation defaults to generic language when retrieval grounding is weak
In the lowest-scored replies (LLM judge score ≤2, n=5), a consistent pattern emerges: when a customer makes a specific, actionable request ("cancel this order," "modify my order"), the generated reply is polite but generically deflects to "check your email" or "contact support" rather than directly engaging with the specific ask.
**Hypothesis:** the retrieval step pulls topically-similar historical examples, but when no closely-matching *specific-action* precedent exists in the retrieved set, the LLM falls back to safe, generic phrasing rather than reasoning about the request directly.

### 5. Reply generation under-answers direct factual questions
For a literal factual question ("Is this feature available in the Amazon India app?"), the generated reply defaulted to "I don't have information on that" rather than attempting a best-effort answer.
**Hypothesis:** the system has no mechanism to distinguish "I should retrieve a resolution pattern" from "I should just answer a direct question" — both get routed through the same RAG-grounded reply pipeline, which is tuned for complaint resolution, not FAQ-style answering.

---

## 4. What Is Misleading About My Headline Number?

The 66.5% intent accuracy and 0.890 escalation F1 look strong in isolation, but four things would mislead a reader who stopped there:

1. **Aggregate accuracy hides highly uneven per-class performance.** The two largest categories tell opposite stories: `delivery_delay` recall is 0.84 (strong), but `product_or_shipping_inquiry` — an equally large category — has recall of only 0.37. A single accuracy number averages over this gap and looks better than the system performs on its second-most-common case.

2. **Escalation recall (0.95) came from a specific engineering fix, and the tradeoff (precision 0.837) is real.** Reporting only the F1 score hides that ~1 in 6 flagged escalations was unnecessary. In a real deployment, that's a direct cost in human agent time.

3. **The golden set's escalation base rate (59.5% marked TRUE) is unusually high for real support traffic**, where routine, non-urgent messages typically dominate. Both the escalation numbers and the baseline comparison are calibrated against this base rate — a golden set with a more realistic ~15-25% escalation rate would likely show a different precision/recall balance for all three systems.

4. **The reply-quality average (3.62/5) is pulled down by a small number of clear failures, not evenly distributed mediocrity.** 5 of 200 replies scored ≤2 — a small fraction, but those are also the cases where a wrong public reply does the most reputational damage. A single average obscures that the failure mode is concentrated, not diffuse.

---

## 5. What I'd Do Next With One More Week

1. **Fix the `customer_discontent` / `product_or_shipping_inquiry` confusion directly** — likely via better few-shot examples in the classification prompt that explicitly contrast tone from topic, or a two-stage classifier (topic first, then a separate escalation-relevant tone signal, decoupled from intent).
2. **Rebuild the golden set's escalation labels with a more realistic base rate**, sampling more routine traffic rather than the current distribution, and re-validate all three systems against it.
3. **Add a lightweight FAQ/direct-question detection path** so factual questions get answered directly instead of routed through complaint-resolution RAG.
4. **Extend to multilingual support**, starting with the next 2-3 highest-volume languages in the raw dataset.
5. **Test on a second brand** to check whether the taxonomy and pipeline generalize, or whether they're overfit to AmazonHelp's specific patterns.
6. **Multi-turn context** — incorporate prior messages in a thread rather than classifying each message in isolation.