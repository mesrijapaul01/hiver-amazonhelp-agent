"""
Builds a stratified candidate list for the golden set from the bootstrapped
(auto-classified) real data. This does NOT replace hand-labeling -- it just
picks a well-distributed sample for you to review and correct, so you're
confirming/fixing labels instead of reading 500 random rows to find enough
of each intent.

Strategy (see eval/labeling_guide.md for the reasoning):
- Roughly even sampling across intents, not proportional to real frequency,
  so rare-but-important intents (account/security, billing, escalation) get
  enough coverage to evaluate meaningfully.
- A handful of low-confidence rows per intent are deliberately included --
  those are exactly the ambiguous/edge cases worth having in the golden set.

Usage:
    python scripts/sample_golden_set.py --input data/historical_threads_bootstrapped.csv --output eval/golden_set_candidates.csv --per-intent 20
"""
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/historical_threads_bootstrapped.csv")
    ap.add_argument("--output", default="eval/golden_set_candidates.csv")
    ap.add_argument("--per-intent", type=int, default=20, help="target rows per intent")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df = df.dropna(subset=["customer_tweet"])

    samples = []
    for intent, group in df.groupby("intent"):
        n = min(args.per_intent, len(group))
        # Mix: half lowest-confidence (the ambiguous/edge cases worth reviewing
        # closely), half random -- so the golden set isn't only "easy" examples.
        group_sorted = group.sort_values("bootstrap_confidence")
        n_low_conf = n // 2
        low_conf_picks = group_sorted.head(n_low_conf)
        remaining = group.drop(low_conf_picks.index)
        random_picks = remaining.sample(min(n - n_low_conf, len(remaining)), random_state=args.seed)
        samples.append(pd.concat([low_conf_picks, random_picks]))

    result = pd.concat(samples).reset_index(drop=True)
    result["id"] = [f"golden_{i:04d}" for i in range(len(result))]

    out = result[["id", "customer_tweet"]].copy()
    out["true_intent"] = result["intent"]  # PRE-FILLED WITH THE BOOTSTRAP GUESS -- review and correct every single one
    out["should_escalate"] = ""
    out["escalate_reason"] = ""
    out["notes"] = ""
    out["_bootstrap_confidence"] = result["bootstrap_confidence"]  # for your reference while reviewing; delete before final submission

    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} candidate rows -> {args.output}")
    print(out["true_intent"].value_counts())
    print(f"\nNEXT STEP: open {args.output} in Excel and go through EVERY row:")
    print("  1. Check 'true_intent' -- the bootstrap classifier pre-filled it, but")
    print("     it WILL be wrong sometimes. Correct it where needed.")
    print("  2. Fill in 'should_escalate' (TRUE/FALSE) based on YOUR judgment, not")
    print("     what the router would say -- this is the ground truth it gets graded against.")
    print("  3. Fill 'escalate_reason' where should_escalate is TRUE.")
    print("  4. Use 'notes' for anything ambiguous, multi-intent, or non-English.")
    print("  5. Delete the '_bootstrap_confidence' column before treating this as final.")


if __name__ == "__main__":
    main()
