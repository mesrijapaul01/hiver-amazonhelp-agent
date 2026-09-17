"""
Samples N rows from the final golden set for a blind self-relabel
consistency check: the output file hides true_intent/should_escalate so you
can relabel from scratch without seeing your original answers, then compare
against eval/golden_set.csv afterward with check_relabel_agreement.py.

Usage:
    python scripts/sample_for_relabel_check.py --input eval/golden_set.csv --output eval/relabel_check.csv --n 15
"""
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="eval/golden_set.csv")
    ap.add_argument("--output", default="eval/relabel_check.csv")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    sample = df.sample(min(args.n, len(df)), random_state=args.seed)

    out = sample[["id", "customer_tweet"]].copy()
    out["relabel_intent"] = ""
    out["relabel_should_escalate"] = ""

    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows for blind relabeling -> {args.output}")
    print("Fill in relabel_intent and relabel_should_escalate WITHOUT looking")
    print("at eval/golden_set.csv for these same rows, then run:")
    print("  python scripts/check_relabel_agreement.py --original eval/golden_set.csv --relabel eval/relabel_check.csv")


if __name__ == "__main__":
    main()
