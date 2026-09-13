"""
Runs any system with a .predict(text) -> {"intent", "escalate", ...} method
against the golden set and reports:
- Intent: accuracy, macro F1, per-class breakdown
- Escalation: precision, recall, F1

USAGE (as a library):
    from eval.metrics import evaluate_system
    from src.baselines.trivial_baseline import TrivialBaseline
    results = evaluate_system(TrivialBaseline(), name="trivial")

USAGE (standalone, scores both baselines):
    python eval/metrics.py
"""
import json
import sys
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report
)

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import GOLDEN_SET_PATH, RESULTS_DIR


def load_golden_set():
    return [json.loads(l) for l in open(GOLDEN_SET_PATH, encoding="utf-8")]


def evaluate_system(system, name: str, verbose: bool = True) -> dict:
    """
    `system` must have a .predict(customer_text) -> dict with keys
    'intent' (str) and 'escalate' (bool). Extra keys ('reply', etc.) ignored here.
    """
    golden = load_golden_set()

    true_intents, pred_intents = [], []
    true_escalate, pred_escalate = [], []
    predictions_log = []

    for row in tqdm(golden, desc=f"Scoring {name}"):
        pred = system.predict(row["customer_text"])
        true_intents.append(row["intent"])
        pred_intents.append(pred["intent"])
        true_escalate.append(row["should_escalate"])
        pred_escalate.append(pred["escalate"])
        predictions_log.append({
            "customer_text": row["customer_text"],
            "true_intent": row["intent"], "pred_intent": pred["intent"],
            "true_escalate": row["should_escalate"], "pred_escalate": pred["escalate"],
            "pred_reply": pred.get("reply", ""),
        })

    intent_acc = accuracy_score(true_intents, pred_intents)
    intent_f1_macro = f1_score(true_intents, pred_intents, average="macro", zero_division=0)

    esc_precision = precision_score(true_escalate, pred_escalate, zero_division=0)
    esc_recall = recall_score(true_escalate, pred_escalate, zero_division=0)
    esc_f1 = f1_score(true_escalate, pred_escalate, zero_division=0)

    results = {
        "system_name": name,
        "n_examples": len(golden),
        "intent_accuracy": round(intent_acc, 4),
        "intent_macro_f1": round(intent_f1_macro, 4),
        "escalation_precision": round(esc_precision, 4),
        "escalation_recall": round(esc_recall, 4),
        "escalation_f1": round(esc_f1, 4),
    }

    if verbose:
        print(f"\n{'='*50}\nSYSTEM: {name}\n{'='*50}")
        print(f"Intent accuracy:      {intent_acc:.1%}")
        print(f"Intent macro F1:      {intent_f1_macro:.4f}")
        print(f"Escalation precision: {esc_precision:.4f}")
        print(f"Escalation recall:    {esc_recall:.4f}")
        print(f"Escalation F1:        {esc_f1:.4f}")
        print(f"\nPer-intent breakdown:")
        print(classification_report(true_intents, pred_intents, zero_division=0))

    out_path = RESULTS_DIR / f"predictions_{name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for p in predictions_log:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    return results


def save_summary(all_results: list[dict]):
    summary_path = RESULTS_DIR / "metrics_summary.json"
    existing = []
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
    # replace any existing entry with the same system_name, else append
    by_name = {r["system_name"]: r for r in existing}
    for r in all_results:
        by_name[r["system_name"]] = r
    summary_path.write_text(json.dumps(list(by_name.values()), indent=2), encoding="utf-8")
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    from src.baselines.trivial_baseline import TrivialBaseline
    from src.baselines.simple_baseline import SimpleBaseline

    trivial_results = evaluate_system(TrivialBaseline(), name="trivial_baseline")
    simple_results = evaluate_system(SimpleBaseline(), name="simple_baseline")

    save_summary([trivial_results, simple_results])

    print(f"\n{'='*50}\nCOMPARISON\n{'='*50}")
    for r in [trivial_results, simple_results]:
        print(f"{r['system_name']:20s} | intent_acc={r['intent_accuracy']:.1%} | "
              f"esc_f1={r['escalation_f1']:.4f}")