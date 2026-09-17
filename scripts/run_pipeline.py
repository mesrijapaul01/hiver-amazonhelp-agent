"""
Run the full agent pipeline over a CSV of new customer messages.

Usage:
    python scripts/run_pipeline.py --input data/new_messages.csv --output outputs/results.csv

Input CSV must have a `customer_tweet` column.
"""
import argparse
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import process_message


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--historical", default="data/historical_threads.csv")
    ap.add_argument("--limit", type=int, default=None, help="only process first N rows (useful for smoke tests)")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)

    results = []
    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] processing...", file=sys.stderr)
        try:
            result = process_message(row["customer_tweet"], historical_csv=args.historical)
        except Exception as e:  # noqa: BLE001
            result = {"customer_tweet": row["customer_tweet"], "error": str(e)}
        results.append(result)

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} results -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
