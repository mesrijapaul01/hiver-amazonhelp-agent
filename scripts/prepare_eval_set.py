"""
Converts eval/golden_set_candidates.csv (the file you hand-labeled) into the
final golden set used for evaluation: drops EXCLUDED_NON_ENGLISH and
DUPLICATE rows, normalizes should_escalate to real booleans, and keeps only
the columns eval/metrics.py and eval/llm_judge.py actually need.

Usage:
    python scripts/prepare_eval_set.py --input eval/golden_set_candidates.csv --output eval/golden_set.csv
"""
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="eval/golden_set_candidates.csv")
    ap.add_argument("--output", default="eval/golden_set.csv")
    args = ap.parse_args()

    # latin1 handles the mixed encoding from Excel edits with non-English text
    df = pd.read_csv(args.input, encoding="latin1")
    df.columns = [c.strip() for c in df.columns]

    before = len(df)
    excluded = df[df["true_intent"].isin(["EXCLUDED_NON_ENGLISH", "DUPLICATE"])]
    df = df[~df["true_intent"].isin(["EXCLUDED_NON_ENGLISH", "DUPLICATE"])].copy()

    # Normalize should_escalate to real booleans (Excel may have saved
    # TRUE/FALSE as strings, or left some blank)
    def to_bool(v):
        if pd.isna(v):
            return False
        return str(v).strip().upper() == "TRUE"

    df["should_escalate"] = df["should_escalate"].apply(to_bool)

    keep_cols = ["id", "customer_tweet", "true_intent", "should_escalate", "escalate_reason", "notes"]
    df = df[[c for c in keep_cols if c in df.columns]]

    df.to_csv(args.output, index=False)
    print(f"Input rows: {before}")
    print(f"Excluded (non-English / duplicate): {len(excluded)}")
    print(f"Final eval set: {len(df)} rows -> {args.output}")
    print()
    print("true_intent distribution in final eval set:")
    print(df["true_intent"].value_counts())
    print()
    print(f"should_escalate=True: {df['should_escalate'].sum()} / {len(df)}")


if __name__ == "__main__":
    main()
