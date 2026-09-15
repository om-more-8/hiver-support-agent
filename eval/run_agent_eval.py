"""
Scores the REAL AI pipeline (intent classifier + retrieval + reply
generation + escalation logic) against the golden set — same scoring
function used for the baselines, so results are directly comparable.

This makes real LLM API calls (~2 per row x 200 rows = ~400 calls).
On gpt-4o-mini this costs roughly a few cents total, but will take a
few minutes to run — that's expected, not a hang.

USAGE:
    python eval/run_agent_eval.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.pipeline import SupportAgent
from eval.metrics import evaluate_system, save_summary
import json
from src.config import RESULTS_DIR


def main():
    agent = SupportAgent()
    results = evaluate_system(agent, name="ai_agent")
    save_summary([results])

    # Print full three-way comparison if baseline results already exist
    summary_path = RESULTS_DIR / "metrics_summary.json"
    all_results = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"\n{'='*60}\nFULL COMPARISON\n{'='*60}")
    for r in all_results:
        print(f"{r['system_name']:20s} | intent_acc={r['intent_accuracy']:.1%} | "
              f"intent_f1={r['intent_macro_f1']:.4f} | esc_f1={r['escalation_f1']:.4f}")


if __name__ == "__main__":
    main()