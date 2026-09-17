"""
Filters the real Kaggle 'Customer Support on Twitter' dataset (twcs.csv) down
to AmazonHelp reply pairs, and writes data/historical_threads.csv in the same
schema the pipeline already expects -- so nothing else in the repo needs to
change.

Usage:
    python scripts/filter_amazonhelp.py --input "C:\\Users\\srija\\kaggle_data\\twcs\\twcs.csv" --output data/historical_threads.csv --limit 3000

--limit caps how many AmazonHelp reply pairs to keep (the assignment says a
subsample is expected/encouraged -- no need to process all 3M rows).
"""
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to twcs.csv")
    ap.add_argument("--output", default="data/historical_threads.csv")
    ap.add_argument("--brand", default="AmazonHelp")
    ap.add_argument("--limit", type=int, default=3000, help="max AmazonHelp reply pairs to keep")
    args = ap.parse_args()

    print(f"Loading {args.input} (this is a large file, may take a minute)...")
    df = pd.read_csv(args.input, dtype=str)  # dtype=str avoids tweet_id float-coercion issues
    print(f"Loaded {len(df):,} total rows")

    # Index by tweet_id for fast lookup of "what did the customer originally say"
    df = df.set_index("tweet_id", drop=False)

    brand_replies = df[df["author_id"] == args.brand]
    print(f"Found {len(brand_replies):,} {args.brand} reply tweets")

    if args.limit:
        brand_replies = brand_replies.head(args.limit)

    rows = []
    missing = 0
    for _, reply in brand_replies.iterrows():
        parent_id = reply.get("in_response_to_tweet_id")
        if pd.isna(parent_id) or parent_id not in df.index:
            missing += 1
            continue
        customer_tweet_row = df.loc[parent_id]
        # .loc can return a DataFrame if tweet_id isn't unique -- guard against that
        if isinstance(customer_tweet_row, pd.DataFrame):
            customer_tweet_row = customer_tweet_row.iloc[0]

        if not bool(customer_tweet_row.get("inbound") in ("True", True)):
            # parent wasn't actually an inbound customer message -- skip
            continue

        rows.append({
            "thread_id": f"real_{reply['tweet_id']}",
            "customer_tweet": customer_tweet_row["text"],
            "historical_agent_reply": reply["text"],
            "intent": "",  # left blank -- fill in via labeling, or the classifier can bootstrap it
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df):,} real (customer_tweet, agent_reply) pairs -> {args.output}")
    print(f"({missing:,} {args.brand} replies had no findable parent tweet and were skipped)")
    print("\nNOTE: 'intent' column is blank for all real rows -- these need to be")
    print("hand-labelled (see eval/labeling_guide.md) or bootstrapped by running")
    print("the classifier over them and spot-checking, before this replaces the")
    print("synthetic placeholder data for grounding retrieval.")


if __name__ == "__main__":
    main()
