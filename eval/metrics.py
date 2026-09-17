"""
Computes intent classification and routing metrics against the golden set.

Usage:
    python eval/metrics.py --predictions outputs/results.csv --golden eval/golden_set.csv
"""
import argparse
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


def compute_intent_metrics(golden: pd.DataFrame, preds: pd.DataFrame) -> dict:
    merged = golden.merge(preds, on="customer_tweet", how="inner", suffixes=("_true", "_pred"))
    if len(merged) < len(golden):
        print(f"WARNING: only matched {len(merged)}/{len(golden)} golden examples to predictions "
              f"(tweet text mismatch? re-check the join key)")

    y_true = merged["true_intent"]
    y_pred = merged["predicted_intent"]

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true) | set(y_pred)))
    cm_labels = sorted(set(y_true) | set(y_pred))

    return {
        "n": len(merged),
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "confusion_matrix_labels": cm_labels,
        "merged": merged,
    }


def compute_routing_metrics(golden: pd.DataFrame, preds: pd.DataFrame) -> dict:
    merged = golden.merge(preds, on="customer_tweet", how="inner")
    y_true = merged["should_escalate"].astype(bool)
    y_pred = merged["routing_decision"].eq("escalate")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    # The costly error direction: agent auto-handles something that should've
    # been escalated. Report this specifically, not just aggregate F1.
    false_auto_handles = merged[(y_true == True) & (y_pred == False)]  # noqa: E712

    return {
        "n": len(merged),
        "escalate_precision": precision,
        "escalate_recall": recall,
        "escalate_f1": f1,
        "dangerous_false_auto_handle_count": len(false_auto_handles),
        "dangerous_false_auto_handle_examples": false_auto_handles[
            ["customer_tweet", "true_intent", "escalate_reason"]
        ].to_dict("records"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--golden", required=True)
    args = ap.parse_args()

    golden = pd.read_csv(args.golden)
    preds = pd.read_csv(args.predictions)

    intent_metrics = compute_intent_metrics(golden, preds)
    routing_metrics = compute_routing_metrics(golden, preds)

    print("=" * 60)
    print(f"INTENT CLASSIFICATION (n={intent_metrics['n']})")
    print("=" * 60)
    print(f"Accuracy: {intent_metrics['accuracy']:.3f}")
    print(intent_metrics["classification_report"])

    print("=" * 60)
    print(f"ROUTING (n={routing_metrics['n']})")
    print("=" * 60)
    print(f"Escalate precision: {routing_metrics['escalate_precision']:.3f}")
    print(f"Escalate recall:    {routing_metrics['escalate_recall']:.3f}")
    print(f"Escalate F1:        {routing_metrics['escalate_f1']:.3f}")
    print(f"DANGEROUS false-auto-handles (should've escalated, didn't): "
          f"{routing_metrics['dangerous_false_auto_handle_count']}")
    for ex in routing_metrics["dangerous_false_auto_handle_examples"]:
        print(f"  - [{ex['true_intent']}] {ex['customer_tweet'][:80]}  reason: {ex['escalate_reason']}")


if __name__ == "__main__":
    main()
