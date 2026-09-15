"""
Runs the LLM judge over every reply your real agent generated (from
results/predictions_ai_agent.jsonl, produced by run_agent_eval.py),
then samples a subset for YOU to blind-score, so you can measure
judge-vs-human agreement.

USAGE:
    python eval/run_reply_judging.py
Writes:
    results/reply_quality_scores.jsonl       (all 200, judge scores)
    data/human_judge_sample_for_scoring.csv  (40 rows, YOU fill in 'human_score')
"""
import json
import random
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import RESULTS_DIR, ROOT
from eval.llm_judge import judge_reply

HUMAN_SAMPLE_SIZE = 40
RANDOM_SEED = 42


def main():
    preds_path = RESULTS_DIR / "predictions_ai_agent.jsonl"
    records = [json.loads(l) for l in open(preds_path, encoding="utf-8")]

    print(f"Judging {len(records)} replies...")
    scored = []
    for r in tqdm(records, desc="LLM judging replies"):
        judgment = judge_reply(r["customer_text"], r["pred_reply"])
        scored.append({
            "customer_text": r["customer_text"],
            "generated_reply": r["pred_reply"],
            "llm_score": judgment["score"],
            "llm_rationale": judgment["rationale"],
        })

    out_path = RESULTS_DIR / "reply_quality_scores.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Saved {len(scored)} judged replies to {out_path}")

    valid_scores = [s["llm_score"] for s in scored if s["llm_score"] is not None]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    print(f"Average LLM judge score: {avg_score:.2f} / 5")

    random.seed(RANDOM_SEED)
    sample = random.sample(scored, min(HUMAN_SAMPLE_SIZE, len(scored)))
    df = pd.DataFrame(sample)
    df["human_score"] = ""
    df = df[["customer_text", "generated_reply", "human_score", "llm_score", "llm_rationale"]]

    human_path = ROOT / "data" / "human_judge_sample_for_scoring.csv"
    df.to_csv(human_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(sample)}-row sample for human scoring to {human_path}")
    print("IMPORTANT: score 'human_score' (1-5) WITHOUT looking at 'llm_score' first —")
    print("cover/ignore that column while you score, or your agreement number will be inflated.")
    print("Once done, run: python eval/judge_agreement.py")


if __name__ == "__main__":
    main()