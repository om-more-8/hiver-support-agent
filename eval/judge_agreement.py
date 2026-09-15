"""
Compares your human_score (filled in by hand) against llm_score to
measure how well the LLM judge agrees with human judgment. This is
the "evidence of how well your judge agrees with a human" the brief
explicitly requires.

USAGE (after filling in human_score in the CSV):
    python eval/judge_agreement.py
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import ROOT, RESULTS_DIR


def main():
    path = ROOT / "data" / "human_judge_sample_for_scoring.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")

    df["human_score"] = pd.to_numeric(df["human_score"], errors="coerce")
    missing = df["human_score"].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} rows still missing human_score — fill these in before "
              f"trusting the agreement numbers. Proceeding with the {len(df) - missing} scored rows.")
        df = df.dropna(subset=["human_score"])

    if len(df) < 10:
        print("Too few scored rows to compute meaningful agreement. Score more rows first.")
        return

    human = df["human_score"].astype(int)
    llm = df["llm_score"].astype(int)

    exact_match = (human == llm).mean()
    within_1 = (abs(human - llm) <= 1).mean()
    correlation, p_value = spearmanr(human, llm)
    mean_abs_diff = (human - llm).abs().mean()

    print(f"{'='*50}\nJUDGE-VS-HUMAN AGREEMENT (n={len(df)})\n{'='*50}")
    print(f"Exact match rate:        {exact_match:.1%}")
    print(f"Within-1-point rate:     {within_1:.1%}")
    print(f"Spearman correlation:    {correlation:.3f} (p={p_value:.4f})")
    print(f"Mean absolute difference: {mean_abs_diff:.2f} points")
    print(f"\nHuman avg score: {human.mean():.2f} | LLM avg score: {llm.mean():.2f}")

    df["diff"] = (human - llm).abs()
    worst = df.sort_values("diff", ascending=False).head(5)
    print(f"\n{'='*50}\nBIGGEST DISAGREEMENTS (for failure analysis)\n{'='*50}")
    for _, row in worst.iterrows():
        print(f"\nHuman={int(row['human_score'])} LLM={int(row['llm_score'])} "
              f"(diff={int(row['diff'])})")
        print(f"  Customer: {row['customer_text'][:100]}")
        print(f"  Reply:    {row['generated_reply'][:100]}")
        print(f"  LLM rationale: {row['llm_rationale']}")

    summary = {
        "n_scored": len(df),
        "exact_match_rate": round(exact_match, 4),
        "within_1_point_rate": round(within_1, 4),
        "spearman_correlation": round(correlation, 4),
        "mean_absolute_difference": round(mean_abs_diff, 4),
    }
    out_path = RESULTS_DIR / "judge_agreement_summary.json"
    import json
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()