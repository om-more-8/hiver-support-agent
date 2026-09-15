"""
Prints the already-computed headline results from results/*.json —
instant, no API calls. This is the fast path for verifying results
without re-running the full pipeline (which involves ~600 live LLM
calls and takes ~30 min).

USAGE:
    python eval/show_results.py
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import RESULTS_DIR


def main():
    metrics_path = RESULTS_DIR / "metrics_summary.json"
    agreement_path = RESULTS_DIR / "judge_agreement_summary.json"

    print("=" * 60)
    print("HEADLINE RESULTS (pre-computed, committed to repo)")
    print("=" * 60)

    if metrics_path.exists():
        results = json.loads(metrics_path.read_text(encoding="utf-8"))
        print(f"\n{'System':22s} {'Intent Acc':>12s} {'Intent F1':>12s} {'Escalation F1':>15s}")
        for r in results:
            print(f"{r['system_name']:22s} {r['intent_accuracy']:>11.1%} "
                  f"{r['intent_macro_f1']:>12.4f} {r['escalation_f1']:>15.4f}")
    else:
        print(f"\n{metrics_path} not found — run eval/metrics.py and eval/run_agent_eval.py first.")

    if agreement_path.exists():
        agreement = json.loads(agreement_path.read_text(encoding="utf-8"))
        print(f"\nReply-quality judge vs human agreement (n={agreement['n_scored']}):")
        print(f"  Exact match rate:     {agreement['exact_match_rate']:.1%}")
        print(f"  Within-1-point rate:  {agreement['within_1_point_rate']:.1%}")
        print(f"  Spearman correlation: {agreement['spearman_correlation']:.4f}")
    else:
        print(f"\n{agreement_path} not found — run eval/run_reply_judging.py and eval/judge_agreement.py first.")

    print(f"\nFull methodology and analysis: report/REPORT.md")


if __name__ == "__main__":
    main()