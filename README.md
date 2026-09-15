# AI Support Agent — Hiver SDE Intern Take-Home

AI customer support agent for AmazonHelp (Twitter). Classifies incoming customer
messages into intents, drafts replies grounded in how the brand has historically
resolved similar issues, and decides whether a message should be auto-handled or
escalated to a human — with a stated reason.

**Report:** [`report/REPORT.md`](report/REPORT.md)
**Decision log:** [`decision_log.md`](decision_log.md)

## Results Summary

| System | Intent Accuracy | Intent Macro F1 | Escalation F1 |
|---|---|---|---|
| Trivial baseline | 29.0% | 0.045 | 0.746 |
| Simple baseline (keywords + TF-IDF) | 37.5% | 0.325 | 0.388 |
| **AI agent** | **66.5%** | **0.628** | **0.890** |

Reply quality (LLM-as-judge, n=200): **3.62/5**. Judge-vs-human agreement (n=40): 70% exact match, 100% within-1-point, Spearman correlation 0.689. Full analysis in the report.

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your own API key:
```
OPENROUTER_API_KEY=your_key_here
LLM_PROVIDER=openrouter
LLM_MODEL=openai/gpt-4o-mini
```
(Groq is also supported — set `LLM_PROVIDER=groq` and `GROQ_API_KEY` instead.)

Download the dataset from Kaggle (`thoughtvector/customer-support-on-twitter`) and
place `twcs.csv` at `data/raw/twcs.csv`.

## Reproduce the Headline Results (under 15 minutes)

**Fast path — verify the committed results (~2 minutes):**

This repo includes the actual output files from a full run (`results/metrics_summary.json`,
`results/judge_agreement_summary.json`, `results/predictions_*.jsonl`). To verify the
headline numbers without re-running ~600 live LLM calls:

```bash
python eval/show_results.py
```

**Full path — regenerate everything from scratch (~30-35 minutes, optional):**

This makes real LLM API calls end-to-end and rebuilds the embeddings cache. Only needed
to verify the full pipeline works, not required to confirm the headline numbers above.

```bash
# 1. Reconstruct customer<->brand conversation pairs, filtered to AmazonHelp, English-only (~1-2 min)
python src/data_prep/load_dataset.py

# 2. Build the golden evaluation set from the already-labeled data (instant — golden_set.jsonl is included)
python eval/csv_to_jsonl.py

# 3. Score baselines against the golden set (~10 sec)
python eval/metrics.py

# 4. Score the real AI agent against the golden set (~15-18 min — builds embeddings cache first time, then ~600 LLM calls)
python eval/run_agent_eval.py

# 5. Score reply quality with the LLM judge (~6 min)
python eval/run_reply_judging.py
```

## Run the Agent on a Single Message

```bash
python src/pipeline.py --tweet "my order still hasn't arrived after 2 weeks"
```

## Project Structure

```
src/
  data_prep/       # load, reconstruct threads, English filter
  intents/         # LLM few-shot intent classifier
  retrieval/       # embedding-based retrieval over historical resolved threads
  reply/           # RAG-grounded reply generation
  escalation/      # LLM judgment + keyword safety net for escalation decisions
  baselines/       # trivial and simple (non-LLM) baselines
  pipeline.py      # ties everything together — the actual agent
eval/
  build_golden_set.py, csv_to_jsonl.py   # golden set construction + validation
  metrics.py, run_agent_eval.py           # scoring against golden set
  llm_judge.py, run_reply_judging.py, judge_agreement.py  # reply quality + human agreement
data/
  golden_set.jsonl        # 200 hand-labelled examples (included)
  golden_set_for_labeling.csv, human_judge_sample_for_scoring.csv  # labeling working files
results/
  metrics_summary.json, judge_agreement_summary.json, predictions_*.jsonl
report/
  REPORT.md
decision_log.md
```

## Golden Set — Sampling and Labeling Method

200 examples sampled from ~112k English-filtered AmazonHelp customer messages,
stratified by text-length bucket (short/medium/long) to avoid skewing toward
simple one-line messages, fixed random seed for reproducibility
(`eval/build_golden_set.py`). Labeled by hand (intent, escalation decision,
escalation reason) with AI-assisted draft suggestions reviewed and corrected
before acceptance — see Decision Log, item 6, for why pure LLM auto-labeling
was avoided.
