# AmazonHelp AI Support Agent -- Report

## 1. Problem framing

**What "good" means for this brand.** AmazonHelp handles an extremely high
volume of short, often emotionally charged support requests on a public,
low-friction channel (Twitter). For a brand at this scale, the priority I
optimized for was **minimizing costly wrong auto-handles** (a bot confidently
resolving something that actually needed a human -- especially around money
and account security) over maximizing raw intent-classification accuracy.
A wrong intent label that still gets correctly escalated to a human causes
no real harm; a correct-looking but wrongly auto-handled refund or security
issue does. This shaped the router design (decision #3-4 in decision_log.md):
four intent categories always escalate regardless of classifier confidence,
by design, not because the classifier can't handle them.

**What I chose not to build, and why:**
- **No multi-turn conversation handling** -- the agent scores/responds to a
  single inbound message, not a full thread. Real support conversations are
  multi-turn; handling that is a materially different, harder problem
  involving conversation state, and out of scope for the time available.
- **No fine-tuning** -- used prompted LLM calls only, given the time and data
  volume constraints of a few-day project.
- **No embeddings-based retrieval** -- grounding uses TF-IDF + cosine
  similarity, a deliberately simple choice given how short and formulaic
  AmazonHelp's real replies are (see decision #5).
- **No automated threshold tuning** -- the 0.6 confidence threshold for
  auto-handling is a reasoned starting guess, not empirically tuned against
  the golden set (see decision #8, and section 4 below).

## 2. Results vs. baselines

All numbers below are on the same 154-example hand-labeled golden set
(`eval/golden_set.csv`), sampled from real AmazonHelp Twitter threads.

| | Trivial baseline | Simple (keyword) baseline | Full agent |
|---|---|---|---|
| Intent accuracy | 14.3% | 29.9% | **77.3%** |
| Escalate precision | 44.8% | 49.6% | **85.5%** |
| Escalate recall | 100% | 95.7% | 76.8% |
| Dangerous false-auto-handles | **0** | 3 | 16 |

The trivial baseline always predicts the majority intent (`refund_or_return`)
and always escalates every message. The simple baseline uses keyword matching
per intent, with the same rule-based router as the full agent.

The full agent clearly outperforms both baselines on intent accuracy and
escalation precision -- it is not simply pattern-matching keywords. But the
last row matters more than it first appears: see section 4.

## 3. Failure analysis: top 5 failure modes

**1. The `other_spam_irrelevant` bucket over-absorbs resolved conversations.**
Precision on this intent is only 0.43 (recall 0.91) -- the classifier is
massively over-calling things "spam." Example: `"@AmazonHelp My package just
got here. All good, thank you!"` was classified `other_spam_irrelevant`,
when it's actually a resolved conversation needing no action. Hypothesis:
the 10-intent taxonomy has no category for "issue resolved, no action
needed," so the classifier has nowhere else to put these and defaults to the
closest-sounding bucket.

**2. `general_product_question` is significantly under-triggered.**
Recall is only 0.47 -- real product questions are getting misclassified into
other categories more than half the time. Example: `"@AmazonHelp is there a
way to know if the Amazon gift cards I send are being used?"` -- a genuine
product/feature question -- was one of several such messages pulled toward
adjacent intents instead. Hypothesis: this intent's description in the
classifier prompt overlaps too much with `billing_or_payment_issue` and
`order_status_tracking` in ways that need tightening.

**3. Classification errors cascade into routing failures -- the router
itself is not the primary problem.** Of the 16 dangerous false-auto-handles,
the large majority are cases where the agent predicted a *lower-risk* intent
than the true one (e.g. a `complaint_escalation` message misclassified as
`order_status_tracking`), so the router's "always escalate this category"
rule never had the chance to fire. Example: `"@AmazonHelp Can you please
update over this quickly"` -- labeled `complaint_escalation` in the golden
set, but misclassified by the agent, so it was auto-handled. This is an
important distinction: the routing *policy* is working as designed; the
failures are upstream, in intent classification accuracy on ambiguous or
low-context messages.

**4. Multi-intent messages don't fit the single-label taxonomy cleanly.**
Example: `"@AmazonHelp I have tried. It's not allowing me to access my
account to cancel my membership, I haven't even used prime in almost 3
months."` tangles an account-access problem with a cancellation request.
These were hand-labeled by primary/most-urgent intent (decision #11), but
the classifier doesn't have that same disambiguation guidance, and picks
inconsistently.

**5. Short, low-context tweets (pulled from the middle of a thread) can't be
classified confidently by anyone -- human or model.** Messages like
`"@AmazonHelp Please call me"` or `"@AmazonHelp Yes"` lack the surrounding
conversation that would make their intent clear. This showed up repeatedly
during hand-labeling (at least 10 examples flagged with low-confidence notes
in the golden set) and caps how good intent accuracy can realistically get
on this dataset, independent of model quality -- a meaningful chunk of real
Twitter support data is simply under-specified as standalone messages.

## 4. What is misleading about my headline number?

**77.3% intent accuracy sounds solid, but it is not the number that should
drive a trust decision.** The 16 dangerous false-auto-handles are the number
that matters, and on that metric the *trivial* baseline (which auto-handles
nothing) is strictly safer than the full agent, by construction. A system
that does more will always accumulate more of the costly failure mode unless
its accuracy on exactly the highest-risk category approaches 100% -- and it
doesn't here (`complaint_escalation` recall is only 0.67). Escalation
recall (76.8%) is the more honest headline number than intent accuracy, and
it says the system is not yet safe to deploy unsupervised on high-risk
traffic.

**A real data-leakage risk exists in the grounding retrieval.** The 154
golden-set tweets were sampled from the same 500-row bootstrapped subset of
the historical corpus used for TF-IDF grounding retrieval. This means that
when the agent processes a golden-set tweet, the retriever can find the
*tweet itself* (or a near-duplicate) sitting in the historical corpus as a
"similar past case." This likely inflates measured reply groundedness/quality
relative to how the system would perform on genuinely unseen messages, and
is worth calling out explicitly rather than letting the reply-quality
numbers stand unqualified.

**The retrieval corpus's intent labels were blank at evaluation time.**
`data/historical_threads.csv` (the file the retriever reads from) never had
its `intent` column populated -- only the smaller bootstrapped subset used
for golden-set sampling did. This means the "same-intent filtering" step in
`src/retrieval.py` fell back to whole-corpus similarity search for every
single message during this evaluation run, not the intent-filtered retrieval
the design intended. This is a real implementation gap, not a deliberate
choice, and likely affects grounding quality in ways not yet measured.

**The LLM judge does not reliably agree with independent human scoring.**
I hand-scored 25 replies (sampled to span the judge's full score range, not
just easy high-scoring cases) on the same 4-dimension rubric. Agreement with
the LLM judge, measured as Cohen's kappa (chance-corrected agreement), was
weak across all four dimensions: groundedness 0.14, helpfulness 0.07, tone
-0.01, correctness 0.13 -- all close to zero, meaning the judge's scores
agree with my own judgment barely better than chance would. The judge's
reported mean scores (groundedness 4.26, helpfulness 3.74, tone 4.29,
correctness 4.21, all out of 5) are therefore not a number I'd trust as an
accurate measure of real reply quality without independent human review --
they may reflect the judge's own biases and blind spots more than they
reflect the agent's actual performance. This is arguably the single most
important limitation in this evaluation: the one metric meant to validate
reply quality is itself unvalidated.

**Self-reported LLM confidence is not calibrated.** During bootstrap
classification of 500 real messages, zero rows came back below the 0.6
confidence threshold -- which is itself suspicious, not reassuring. A model
that is never unsure is not a model whose confidence score can be trusted as
a real uncertainty signal, which undercuts the router's confidence-based
escalation logic (decision #8) more than the clean numbers suggest.

**n=154 is a small sample.** Per-intent support counts range from 8
(`billing_or_payment_issue`) to 22 (`refund_or_return`) -- confidence
intervals on the reported precision/recall for the smaller categories are
wide, and a handful of different labeling calls could shift per-intent
numbers noticeably.

**My own labeling is not perfectly self-consistent, and the escalation
judgment is less consistent than intent labeling.** A blind self-relabel
check on 15 randomly sampled golden-set examples found 86.7% intent
agreement with my own original labels, and only 73.3% agreement on
`should_escalate` -- notably lower. The intent disagreements were both
genuinely ambiguous boundary cases (consistent with failure mode #4 above),
which is a real finding about the taxonomy, not just noise. The lower
escalation agreement is more concerning: `should_escalate` is the exact
ground truth the router's highest-stakes behavior is graded against, and if
I don't consistently agree with myself on it, the escalation precision/recall
numbers in section 2 carry real uncertainty beyond what the point estimates
suggest.

**The golden set's intent distribution does not reflect real traffic.** It
was deliberately stratified to oversample rare high-risk intents (decision
#9), so the 77.3% headline accuracy is not what accuracy on the brand's
actual, real-world message mix would look like -- it is accuracy on a
distribution intentionally weighted toward the hard, high-stakes cases.

## 5. What I'd do next with one more week

- Fix the taxonomy gap directly: add an intent category for resolved/no-action
  messages, and tighten the `general_product_question` description to reduce
  overlap with adjacent intents -- then re-run the full eval to see whether
  intent accuracy (and consequently escalation recall) genuinely improves.
- Fix the retrieval corpus intent-population gap so same-intent filtering
  actually runs as designed, and separately, build a non-overlapping
  historical corpus / golden-set split to eliminate the data-leakage risk
  identified above.
- Sweep the confidence threshold (`CONFIDENCE_ESCALATE_THRESHOLD`) against
  the golden set's actual false-auto-handle rate rather than using an
  unvalidated guess.
- Get a second human labeler on a subset of the golden set and compute real
  inter-annotator agreement -- a solo self-relabel check was done (86.7%
  intent / 73.3% escalate self-agreement, see section 4), but a genuinely
  independent second labeler is the real next step, especially given how
  much lower the escalation agreement was than intent agreement.
- Investigate and fix the LLM judge's weak agreement with human scoring --
  likely next steps: revise the judge rubric prompt to be more specific and
  less lenient, try a different/larger judge model, or score on a coarser
  scale (e.g. pass/fail per dimension instead of 1-5) where agreement may be
  easier to achieve.
- Add a trained (not keyword-matching) ML baseline to more fairly answer
  "how much is the LLM actually buying over a cheap, non-LLM classifier."
- Compare TF-IDF retrieval against an embeddings-based retriever head-to-head
  on reply groundedness, rather than assuming TF-IDF is sufficient.
- Expand the golden set, particularly for the lowest-support intents
  (`billing_or_payment_issue` at n=8), to tighten the confidence intervals
  on per-intent metrics.
