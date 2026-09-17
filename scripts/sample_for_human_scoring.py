"""
Samples rows from the judged output for you to hand-score, deliberately
mixing high, medium, and low judge scores -- agreement on an all-5s sample
tells you nothing. Produces a template you fill in by hand.

Usage:
    python scripts/sample_for_human_scoring.py --input outputs/judged.csv --output eval/human_scores_template.csv --n 25
"""
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/judged.csv")
    ap.add_argument("--output", default="eval/human_scores_template.csv")
    ap.add_argument("--n", type=int, default=25)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    # Use the average of the 4 judge dims to bucket low/mid/high, so the
    # sample spans the judge's actual range rather than clustering at 5s.
    score_cols = ["judge_groundedness", "judge_helpfulness", "judge_tone", "judge_correctness"]
    df["_avg_judge"] = df[score_cols].mean(axis=1)
    df = df.dropna(subset=["_avg_judge"])

    df_sorted = df.sort_values("_avg_judge")
    n = min(args.n, len(df_sorted))
    third = n // 3
    low = df_sorted.head(third)
    high = df_sorted.tail(third)
    remaining = df_sorted.drop(low.index).drop(high.index)
    mid = remaining.sample(min(n - 2 * third, len(remaining)), random_state=7)

    sample = pd.concat([low, mid, high]).sample(frac=1, random_state=7).reset_index(drop=True)  # shuffle order

    out = sample[["customer_tweet", "predicted_intent", "draft_reply"]].copy()
    out["human_groundedness"] = ""
    out["human_helpfulness"] = ""
    out["human_tone"] = ""
    out["human_correctness"] = ""

    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to score by hand -> {args.output}")
    print("Score each on the SAME rubric as eval/llm_judge.py's JUDGE_SYSTEM prompt:")
    print("  groundedness, helpfulness, tone, correctness -- each 1(bad) to 5(excellent)")
    print("Fill in human_groundedness / human_helpfulness / human_tone / human_correctness, then run:")
    print("  python eval/llm_judge.py --agreement outputs/judged.csv --human eval/human_scores_template.csv")


if __name__ == "__main__":
    main()
