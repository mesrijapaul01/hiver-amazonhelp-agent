"""
Bootstraps intent labels on the real historical_threads.csv by running the
LLM classifier over every row. This is NOT a substitute for the hand-labelled
golden set -- it's a starting point so you're correcting labels instead of
writing all 2,000+ from scratch, and so the grounding retrieval in src/retrieval.py
has real intent labels to filter on instead of falling back to un-filtered
similarity search.

This decision (bootstrap + spot-check vs. fully manual) should go in
decision_log.md -- it trades some label noise for a lot of time saved, and
that trade-off is exactly the kind of thing worth being explicit about.

Usage:
    python scripts/bootstrap_intents.py --input data/historical_threads.csv --output data/historical_threads.csv --limit 500
"""
import argparse
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.classify_intent import classify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=None, help="only label the first N rows (cheaper to test with)")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit).copy()

    intents = []
    confidences = []
    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] classifying...", file=sys.stderr)
        try:
            result = classify(row["customer_tweet"])
            intents.append(result["intent"])
            confidences.append(result["confidence"])
        except Exception as e:  # noqa: BLE001
            print(f"  error on row {i}: {e}", file=sys.stderr)
            intents.append("")
            confidences.append(0.0)

    df["intent"] = intents
    df["bootstrap_confidence"] = confidences
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} bootstrapped rows -> {args.output}", file=sys.stderr)
    print(df["intent"].value_counts())
    print(f"\nLow-confidence rows (confidence < 0.6): {(df['bootstrap_confidence'] < 0.6).sum()} "
          f"-- these are the ones most worth checking by hand first.")


if __name__ == "__main__":
    main()
