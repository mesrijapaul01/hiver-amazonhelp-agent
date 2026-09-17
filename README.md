# AmazonHelp AI Support Agent

An AI agent that classifies customer support tweets into intents, drafts a
reply grounded in how AmazonHelp has historically resolved similar issues,
and decides whether to auto-handle or escalate to a human -- with a stated
reason for every decision.

Built for the Hiver SDE Intern take-home assignment.

**Headline results** (154-example hand-labeled golden set): 77.3% intent
accuracy, 85.5% escalation precision, 76.8% escalation recall, vs. 14.3% /
44.8% / 100% for a trivial always-escalate baseline. See `report.md` for the
full results table, failure analysis, and -- importantly -- why these
numbers should not be taken at face value on their own.

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
`openai/gpt-oss-120b` for drafting/judging -- see `src/config.py`). No
credit card required for a Groq account.

## Reproduce headline results (<15 min)

The golden set (`eval/golden_set.csv`, 154 real AmazonHelp tweets) and the
historical grounding corpus (`data/historical_threads.csv`, ~2,984 real
customer/reply pairs filtered from the Kaggle dataset) are already included
in this repo, so you don't need to re-download or re-filter anything to
reproduce the headline numbers.

```bash
# 1. Run the agent over the golden set
python scripts/run_pipeline.py \
    --input eval/golden_set.csv \
    --output outputs/agent_results.csv

# 2. Intent + routing metrics against ground truth
python eval/metrics.py \
    --predictions outputs/agent_results.csv \
    --golden eval/golden_set.csv

# 3. Baselines, for comparison
python scripts/run_baseline.py --input eval/golden_set.csv --output outputs/trivial_results.csv --baseline trivial
python scripts/run_baseline.py --input eval/golden_set.csv --output outputs/simple_results.csv --baseline simple
python eval/metrics.py --predictions outputs/trivial_results.csv --golden eval/golden_set.csv
python eval/metrics.py --predictions outputs/simple_results.csv --golden eval/golden_set.csv

# 4. LLM-judge reply quality
python eval/llm_judge.py --predictions outputs/agent_results.csv --out outputs/judged.csv

# 5. Judge-human agreement check (a pre-filled 25-example human-scored
#    sample is included at eval/human_scores_template.csv; to redo it
#    yourself, regenerate a fresh sample first)
python scripts/sample_for_human_scoring.py --input outputs/judged.csv --output eval/human_scores_template.csv --n 25
# ... hand-score the 25 rows using the same rubric as eval/llm_judge.py's JUDGE_SYSTEM prompt, then:
python eval/llm_judge.py --agreement outputs/judged.csv --human eval/human_scores_template.csv
```

Steps 1-4 run in well under 15 minutes (mostly LLM API latency, ~154 calls
each for steps 1 and 4; steps 3 are instant, no API calls). Step 5's
hand-scoring is a manual step, not something the 15-minute reproduction
window is meant to cover -- a completed sample is already included so the
agreement numbers in `report.md` can be reproduced by running the last
command alone.

## Rebuilding the golden set / historical corpus from scratch (optional)

The repo already includes the filtered real data and the final hand-labeled
golden set, but if you want to rebuild either from the raw Kaggle export:

```bash
# 1. Download thoughtvector/customer-support-on-twitter from Kaggle, get twcs.csv

# 2. Filter to AmazonHelp (customer_tweet, historical_agent_reply) pairs
python scripts/filter_amazonhelp.py --input path/to/twcs.csv --output data/historical_threads.csv --limit 3000

# 3. Bootstrap intent labels with the classifier (starting point, not ground truth)
python scripts/bootstrap_intents.py --input data/historical_threads.csv --output data/historical_threads_bootstrapped.csv --limit 500

# 4. Stratified-sample candidates for hand-labeling (see eval/labeling_guide.md)
python scripts/sample_golden_set.py --input data/historical_threads_bootstrapped.csv --output eval/golden_set_candidates.csv --per-intent 20

# 5. Hand-label eval/golden_set_candidates.csv (the real, time-consuming step --
#    correct intents, set should_escalate + reasons, mark EXCLUDED_NON_ENGLISH /
#    DUPLICATE as needed), then finalize:
python scripts/prepare_eval_set.py --input eval/golden_set_candidates.csv --output eval/golden_set.csv
```

## Repo layout

```
src/
  config.py            intent taxonomy, risk tiers, model names
  llm_client.py         Groq API wrapper
  classify_intent.py    intent classification
  retrieval.py           TF-IDF grounding retrieval
  draft_reply.py         reply drafting
  router.py              auto-handle / escalate rule
  pipeline.py             orchestration (single source of truth)
  baselines.py            trivial + simple baselines
eval/
  golden_set.csv                final 154-example hand-labeled eval set
  golden_set_candidates.csv      pre-final labeling working file (195 rows)
  human_scores_template.csv       25-example human-scored sample for judge agreement
  labeling_guide.md                how the golden set was sampled & labeled
  metrics.py                        intent accuracy, routing precision/recall
  llm_judge.py                       LLM-as-judge rubric + human-agreement check
scripts/
  filter_amazonhelp.py          extracts real AmazonHelp pairs from raw Kaggle data
  bootstrap_intents.py           auto-labels intents on real data as a labeling starting point
  sample_golden_set.py            stratified sampling for golden-set candidates
  prepare_eval_set.py              finalizes the hand-labeled golden set
  run_pipeline.py                   CLI to run the full agent over a CSV
  run_baseline.py                    CLI to run the trivial/simple baselines
  sample_for_human_scoring.py         samples judge output for hand-scoring
  generate_placeholder_data.py         builds synthetic dev data (not used in final results)
data/
  historical_threads.csv        real grounding corpus (~2,984 AmazonHelp pairs)
  historical_threads_bootstrapped.csv   500-row auto-labeled subset used for golden-set sampling
report.md               problem framing, results, failure analysis, next steps
decision_log.md           15 non-obvious decisions and why

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
