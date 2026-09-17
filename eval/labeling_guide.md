# Golden Set: Sampling & Labeling Guide

**What was actually built: 195 candidate examples sampled, 154 usable in the
final golden set** (`eval/golden_set.csv`), within the assignment's
150-250 target range.

## Sampling process actually used

1. Filtered the raw Kaggle dataset (2,811,774 total tweets) to AmazonHelp
   reply pairs using `scripts/filter_amazonhelp.py`, reconstructing
   (customer inbound message, AmazonHelp's historical reply) pairs via
   `in_response_to_tweet_id` -- 169,840 AmazonHelp replies found, capped to
   a 2,984-row subsample (`data/historical_threads.csv`).
2. Bootstrap-classified a 500-row subset of that real data using the LLM
   classifier (`scripts/bootstrap_intents.py`) -- an auto-labeling starting
   point to make stratified sampling possible, NOT treated as ground truth.
3. Stratified-sampled ~20 candidates per intent from the bootstrapped set
   (`scripts/sample_golden_set.py`), deliberately oversampling rare
   high-risk intents (account/security, billing, escalation) relative to
   their real frequency, so the eval could say something meaningful about
   them -- at the cost of the golden set not reflecting true traffic mix
   (see decision_log.md #9). This produced 195 candidates.
4. Within each intent bucket, half the sample was deliberately drawn from
   the classifier's *lowest-confidence* predictions (the ambiguous/edge
   cases worth reviewing closely), and half was random -- so the golden set
   isn't only "easy" examples.

## Labeling process actually used

Every one of the 195 candidates was individually hand-reviewed:
- `true_intent`: the bootstrap classifier's guess was corrected wherever
  wrong, checked against the fixed taxonomy in `src/config.py`.
- `should_escalate`: set by genuine judgment on whether the message needs a
  human, not by what the router's rule would output -- since this is the
  ground truth the router gets graded against. All examples in the four
  hard-coded-escalate intent categories (`refund_or_return`,
  `billing_or_payment_issue`, `account_or_login_issue`,
  `complaint_escalation`) were set to `should_escalate=TRUE`, matching the
  router's actual policy (decision #4).
- `escalate_reason`: filled for every `should_escalate=TRUE` row.
- `notes`: used for ambiguous, multi-intent, or non-English cases, and for
  messages too short/context-free to classify confidently.
- Non-English tweets (Japanese, Spanish, Portuguese, German, Hindi found in
  the sample) were marked `EXCLUDED_NON_ENGLISH` rather than deleted,
  preserving a visible record of what was excluded (decision #15). Exact
  duplicate tweets were marked `DUPLICATE`. 41 of 195 candidates were
  excluded this way, leaving the final 154-example golden set.

## Label quality check

A blind self-relabel check was performed: 15 examples were randomly sampled
from the final golden set and relabeled from scratch (intent + escalate
decision only, no access to the original labels), then compared against the
original labels.

- **Intent self-agreement: 86.7% (13/15).** Below the 90% target noted
  above -- the two disagreements were both genuinely ambiguous boundary
  cases (`order_status_tracking` vs. `delivery_delay_or_lost` for a message
  about a slipping delivery date; `refund_or_return` vs.
  `complaint_escalation` for an angry, unresolved 40-day complaint with an
  implicit refund ask). This is consistent with failure mode #4 in
  report.md (multi-intent/ambiguous messages don't fit the single-label
  taxonomy cleanly) -- the inconsistency reflects genuine ambiguity in the
  data and taxonomy boundaries, not careless labeling.
- **should_escalate self-agreement: 73.3% (11/15) -- noticeably lower than
  intent agreement.** This is a real finding in its own right: the
  escalate/don't-escalate judgment call is less consistent than intent
  labeling itself, even though should_escalate is the ground truth the
  router's most consequential behavior gets graded against. See
  report.md's "misleading headline number" section.
