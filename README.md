# AmazonHelp AI Support Agent

An AI agent that classifies customer support tweets into intents, drafts a
reply grounded in how AmazonHelp has historically resolved similar issues,
and decides whether to auto-handle or escalate to a human -- with a stated
reason for every decision.

Built for the Hiver SDE Intern take-home assignment.

## Status / what's real vs placeholder

**`data/historical_threads.csv` currently contains 50 SYNTHETIC example
threads I wrote by hand, not the real Kaggle dataset.** This was deliberate:
it let me build and test the full pipeline immediately without waiting on a
Kaggle download inside a sandboxed environment. Before generating the
headline numbers in `report.md`, this file needs to be replaced with a real
subsample -- see "Using the real dataset" below.

## Architecture

```
customer tweet
      |
      v
[classify_intent]  -- LLM classifies into one of 10 intents (src/config.py)
      |
      v
[retrieval]         -- TF-IDF retrieval of similar historically-resolved
      |                 threads with the same intent (src/retrieval.py)
      v
[draft_reply]       -- LLM drafts a reply grounded in the retrieved examples,
      |                 instructed not to invent policy it can't support
      v
[router]            -- rule-based: risk tier of the intent + classifier
                        confidence + whether grounding was found
                        -> auto_handle or escalate, with a reason
```

See `src/pipeline.py` for the orchestration -- one code path used by both
the eval harness and the CLI runner, so "what the agent does" is defined
in exactly one place.

## Setup (5 min)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...   # free key: https://console.groq.com/keys
```

Uses Groq's free tier (`openai/gpt-oss-20b` for classification/routing,
`openai/gpt-oss-120b` for drafting/judging -- see `src/config.py`). No credit
card required for a Groq account.

## Reproduce headline results (<15 min)

```bash
# 1. Run the agent over the golden set's tweets
python scripts/run_pipeline.py \
    --input eval/golden_set.csv \
    --output outputs/results.csv

# 2. Intent + routing metrics against ground truth
python eval/metrics.py \
    --predictions outputs/results.csv \
    --golden eval/golden_set.csv

# 3. LLM-judge reply quality
python eval/llm_judge.py \
    --predictions outputs/results.csv \
    --out outputs/judged.csv

# 4. (separately, offline) score ~30-40 of the same replies yourself using
#    the same rubric in eval/llm_judge.py's JUDGE_SYSTEM prompt, save to
#    eval/human_scores.csv, then:
python eval/llm_judge.py --agreement outputs/judged.csv --human eval/human_scores.csv

# 5. Baselines, for comparison (see src/baselines.py + report.md)
```

With the 50-example placeholder set and a ~150-200 example golden set, steps
1-3 run in well under 15 minutes (mostly LLM API latency).

## Using the real dataset

1. Download `thoughtvector/customer-support-on-twitter` from Kaggle.
2. Filter to `author_id == "AmazonHelp"` and reconstruct (customer inbound
   message, AmazonHelp's reply) pairs using `in_response_to_tweet_id`.
3. Replace `data/historical_threads.csv` with a real subsample (a few
   thousand rows is plenty -- see the assignment's "we will not run this on
   the full dataset" note). Keep the same column schema:
   `thread_id, customer_tweet, historical_agent_reply, intent` -- you'll need
   to hand-label `intent` for a subset, or bootstrap it from the classifier
   and spot-check.
4. Re-run the golden-set sampling & labeling process in
   `eval/labeling_guide.md` against the real data.
5. `MAJORITY_INTENT` in `src/baselines.py` should be updated to match the
   real distribution.

## Repo layout

```
src/
  config.py          intent taxonomy, risk tiers, model names
  llm_client.py       Groq API wrapper
  classify_intent.py  intent classification
  retrieval.py         TF-IDF grounding retrieval
  draft_reply.py       reply drafting
  router.py            auto-handle / escalate rule
  pipeline.py           orchestration (single source of truth)
  baselines.py          trivial + simple baselines
eval/
  golden_set_template.csv   columns for the hand-labelled golden set
  labeling_guide.md          how the golden set was/should be sampled & labeled
  metrics.py                  intent accuracy, routing precision/recall
  llm_judge.py                 LLM-as-judge rubric + human-agreement check
scripts/
  generate_placeholder_data.py   builds the synthetic historical_threads.csv
  run_pipeline.py                  CLI to run the agent over a CSV
data/
  historical_threads.csv     grounding corpus (currently synthetic -- see above)
report.md               problem framing, results, failure analysis, next steps
decision_log.md           non-obvious decisions and why
```

## Citations / borrowed material

- Dataset: Kaggle `thoughtvector/customer-support-on-twitter`.
- No external code borrowed beyond standard library usage of
  `scikit-learn` (TF-IDF, cosine similarity, sklearn.metrics) and the
  `groq` Python SDK, both used per their public docs.
- Built with AI coding assistance (Claude, Anthropic) throughout -- used for
  code implementation, debugging, and drafting report sections based on
  results I generated and reviewed. I can explain and modify any part of
  this codebase.
