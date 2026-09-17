"""
Runs the trivial and simple keyword baselines (src/baselines.py) over a CSV
of messages, producing output in the same schema as run_pipeline.py so
eval/metrics.py can compare all three (trivial / simple / full agent) with
the same script.

No API calls -- these baselines are instant and free, unlike the full agent.

Usage:
    python scripts/run_baseline.py --input eval/golden_set.csv --output outputs/trivial_results.csv --baseline trivial
    python scripts/run_baseline.py --input eval/golden_set.csv --output outputs/simple_results.csv --baseline simple
"""
import argparse
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.baselines import trivial_predict, simple_keyword_predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--baseline", choices=["trivial", "simple"], required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    predict_fn = trivial_predict if args.baseline == "trivial" else simple_keyword_predict

    results = []
    for _, row in df.iterrows():
        pred = predict_fn(row["customer_tweet"])
        results.append({
            "customer_tweet": row["customer_tweet"],
            "predicted_intent": pred["predicted_intent"],
            "draft_reply": pred["draft_reply"],
            "routing_decision": pred["routing_decision"],
            "routing_reason": pred["routing_reason"],
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} {args.baseline} baseline results -> {args.output}")


if __name__ == "__main__":
    main()
