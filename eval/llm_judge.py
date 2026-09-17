"""
LLM-as-judge rubric for reply quality, and a human-agreement check.

Why a rubric with 4 sub-scores instead of one "is this good? 1-5": a single
holistic score lets the judge average away a serious groundedness failure
against good tone. We want groundedness failures to be visible even when the
reply "reads" fine.

Usage:
    python eval/llm_judge.py --predictions outputs/results.csv --golden eval/golden_set.csv --out outputs/judged.csv

Then, separately, hand-score a random ~30-40 of the same replies yourself
(same rubric) and run:
    python eval/llm_judge.py --agreement outputs/judged.csv --human eval/human_scores.csv
"""
import argparse
import sys
import os
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import JUDGE_MODEL
from src.llm_client import call_llm_json

JUDGE_SYSTEM = """You are grading a draft customer-support reply for quality. Score each dimension
1 (bad) to 5 (excellent). Be strict — a 3 is "acceptable but flawed", not "fine".

Dimensions:
- groundedness: Does the reply avoid inventing policies, refund amounts, or timelines
  not supported by the provided context? (5 = fully grounded / appropriately asks for
  more info; 1 = confidently makes up specifics)
- helpfulness: Does it actually address the customer's problem or move it forward?
- tone: Is it appropriately empathetic and professional for the situation?
- correctness: Given the intent, is this a sensible resolution path (not necessarily
  the only right one, but not obviously wrong)?

Respond with ONLY a JSON object:
{"groundedness": <1-5>, "helpfulness": <1-5>, "tone": <1-5>, "correctness": <1-5>, "rationale": "<one sentence>"}"""


def judge_reply(tweet: str, intent: str, reply: str) -> dict:
    if not reply or not reply.strip():
        return {"groundedness": None, "helpfulness": 1, "tone": None, "correctness": 1,
                 "rationale": "Empty reply (likely an escalated/no-draft case)."}
    user = f"""Customer message: \"\"\"{tweet}\"\"\"
Predicted intent: {intent}
Draft reply: \"\"\"{reply}\"\"\""""
    return call_llm_json(model=JUDGE_MODEL, system=JUDGE_SYSTEM, user=user, max_tokens=250)


def run_judge(predictions_path: str, out_path: str):
    df = pd.read_csv(predictions_path)
    scores = []
    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] judging...", file=sys.stderr)
        try:
            s = judge_reply(row["customer_tweet"], row.get("predicted_intent", ""), row.get("draft_reply", ""))
        except Exception as e:  # noqa: BLE001
            s = {"error": str(e)}
        scores.append(s)
    scored = pd.concat([df, pd.DataFrame(scores).add_prefix("judge_")], axis=1)
    scored.to_csv(out_path, index=False)
    print(f"Wrote {len(scored)} judged rows -> {out_path}", file=sys.stderr)

    numeric_cols = [c for c in scored.columns if c.startswith("judge_") and c != "judge_rationale" and c != "judge_error"]
    for c in numeric_cols:
        vals = pd.to_numeric(scored[c], errors="coerce").dropna()
        if len(vals):
            print(f"{c}: mean={vals.mean():.2f}  n={len(vals)}")


def cohen_kappa(a: list, b: list) -> float:
    """Minimal Cohen's kappa for ordinal 1-5 scores treated as categorical agreement."""
    from collections import Counter
    n = len(a)
    assert n == len(b) and n > 0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    cats = set(a) | set(b)
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def _normalize_for_match(s) -> str:
    """Strip whitespace, punctuation, and non-ASCII noise so rows still match
    even if Excel re-encoded special characters (curly quotes, em-dashes,
    apostrophes) differently on save than how they originally came back from
    the API."""
    if pd.isna(s):
        return ""
    ascii_only = "".join(ch for ch in str(s) if ch.isascii() and (ch.isalnum() or ch.isspace()))
    return " ".join(ascii_only.split()).strip().lower()[:60]


def run_agreement(judged_path: str, human_path: str):
    judged = pd.read_csv(judged_path)
    # Excel/WPS often saves with Windows-1252 characters (em-dashes, curly
    # quotes) that aren't valid UTF-8 -- read leniently rather than crash.
    try:
        human = pd.read_csv(human_path)
    except UnicodeDecodeError:
        human = pd.read_csv(human_path, encoding="latin1")
    # expects: customer_tweet, human_groundedness, human_helpfulness, human_tone, human_correctness

    judged["_match_key"] = judged["customer_tweet"].apply(_normalize_for_match)
    human["_match_key"] = human["customer_tweet"].apply(_normalize_for_match)
    exact_matches = len(judged.merge(human, on="customer_tweet", how="inner"))
    merged = judged.merge(human, on="_match_key", how="inner", suffixes=("", "_human"))
    if len(merged) < len(human):
        print(f"NOTE: matched {len(merged)}/{len(human)} human-scored rows "
              f"({exact_matches} matched on exact text) -- some rows may have "
              f"been altered when saving in Excel.")
    print(f"Comparing on n={len(merged)} jointly-scored examples")
    for dim in ["groundedness", "helpfulness", "tone", "correctness"]:
        j = pd.to_numeric(merged[f"judge_{dim}"], errors="coerce")
        h = pd.to_numeric(merged[f"human_{dim}"], errors="coerce")
        valid = j.notna() & h.notna()
        if valid.sum() == 0:
            continue
        kappa = cohen_kappa(j[valid].round().astype(int).tolist(), h[valid].round().astype(int).tolist())
        exact_match = (j[valid].round() == h[valid].round()).mean()
        mae = (j[valid] - h[valid]).abs().mean()
        print(f"{dim}: n={valid.sum()}  exact_match={exact_match:.2f}  MAE={mae:.2f}  Cohen's kappa={kappa:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions")
    ap.add_argument("--out")
    ap.add_argument("--agreement", help="path to judged CSV, to run agreement check instead")
    ap.add_argument("--human", help="path to human_scores.csv, used with --agreement")
    args = ap.parse_args()

    if args.agreement:
        run_agreement(args.agreement, args.human)
    else:
        run_judge(args.predictions, args.out)


if __name__ == "__main__":
    main()
