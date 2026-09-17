# Golden Set: Sampling & Labeling Guide

**Target: 150-250 hand-labelled examples.** Fill `golden_set_template.csv`.

## Sampling strategy (fill in the real numbers once you've pulled the Kaggle subsample)

Don't just take the first N rows for AmazonHelp — that oversamples whatever
issue happened to be trending that week and undersamples rare-but-important
cases (account security, angry escalations). Recommended approach:

1. Filter the Kaggle dataset to `author_id == "AmazonHelp"` inbound-reply pairs
   (i.e. keep the *customer's first message in a thread* + note whether
   AmazonHelp's historical reply resolved it).
2. Stratify: aim for roughly even coverage across your 8-10 intents, not
   proportional-to-frequency — you specifically want enough examples of the
   rare high-risk intents (account/security, billing, escalations) to
   evaluate them meaningfully, even though they're a minority of real traffic.
   A reasonable split for ~200 examples: ~15-25 per intent, with a few extra
   for `complaint_escalation` and `other_spam_irrelevant` since those are the
   ones a bad agent gets wrong in the most costly ways.
3. Within each intent bucket, sample randomly (not "cherry-pick clean ones") —
   include some genuinely ambiguous/borderline messages, since that's where
   real disagreement and real risk live.
4. Explicitly include a handful (~10-15) of adversarial/edge cases: sarcasm,
   multi-intent messages ("also cancel my prime while you're at it"),
   non-English or code-mixed text if present in the data, and messages with
   no clear resolution in history (tests the "don't invent a policy" rule).

## Labeling process

For each sampled tweet, a human (you) labels:
- `true_intent`: pick from the fixed taxonomy in `src/config.py`. If truly
  none fit, use `other_spam_irrelevant` and note why in `notes`.
- `should_escalate`: your judgment call on whether this genuinely needs a
  human (not just "what the rule-based router would say" — you're building
  the ground truth the router gets *graded against*).
- `escalate_reason`: one line, if `should_escalate` is true.
- `notes`: anything ambiguous, multi-intent, or that a second labeler might
  disagree on.

## Label quality check (do this even solo)

Re-label a random 20-30 of your own examples ~a day later without looking at
your first pass. Compute agreement with yourself (simple % match on
`true_intent` and `should_escalate`). Report this number in the report's
"what's misleading about my headline number" section — if you can't agree
with yourself 90%+ of the time, the taxonomy or the data is genuinely
ambiguous, and that's a real finding, not a labeling failure to hide.
