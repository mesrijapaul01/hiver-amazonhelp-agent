"""
Compares a blind relabel pass (from sample_for_relabel_check.py) against the
original golden set labels for the same rows, and reports self-agreement --
this is the "label quality check" described in eval/labeling_guide.md.

Usage:
    python scripts/check_relabel_agreement.py --original eval/golden_set.csv --relabel eval/relabel_check.csv
"""
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", default="eval/golden_set.csv")
    ap.add_argument("--relabel", default="eval/relabel_check.csv")
    args = ap.parse_args()

    original = pd.read_csv(args.original)
    relabel = pd.read_csv(args.relabel)

    merged = original.merge(relabel, on="id", suffixes=("_orig", "_relabel"))
    print(f"Comparing on n={len(merged)} rows")

    def norm_intent(v):
        return str(v).strip() if pd.notna(v) else v

    orig_intent = merged["true_intent"].apply(norm_intent)
    relabel_intent = merged["relabel_intent"].apply(norm_intent)
    intent_match = (orig_intent == relabel_intent).mean()
    print(f"\nIntent self-agreement: {intent_match:.1%}")

    disagreements = merged[orig_intent != relabel_intent]
    if len(disagreements):
        print("\nDisagreements:")
        for idx, row in disagreements.iterrows():
            print(f"  [{row['id']}] original='{orig_intent[idx]}' vs relabel='{relabel_intent[idx]}'")
            print(f"      tweet: {row['customer_tweet_orig'][:100]}")

    def to_bool(v):
        if pd.isna(v):
            return None
        s = str(v).strip().upper()
        if s in ("TRUE", "1"):
            return True
        if s in ("FALSE", "0"):
            return False
        return None

    orig_bool = merged["should_escalate"].apply(lambda v: str(v).strip().upper() == "TRUE")
    relabel_bool = merged["relabel_should_escalate"].apply(to_bool)
    valid = relabel_bool.notna()
    if valid.sum():
        escalate_match = (orig_bool[valid] == relabel_bool[valid]).mean()
        print(f"\nshould_escalate self-agreement (n={valid.sum()}): {escalate_match:.1%}")


if __name__ == "__main__":
    main()
