# Decision Log

Non-obvious decisions made while building this, and why.

1. **Chose AmazonHelp over other brands.** AmazonHelp is the 2nd-highest-volume
   brand in the dataset (after AppleSupport), giving enough real conversation
   volume to build a meaningful historical grounding corpus, and its issue
   mix (orders, refunds, account security, billing) covers a genuinely broad
   range of support scenarios rather than one narrow product category.

2. **Intent taxonomy has 10 categories, not more.** More granularity (e.g.
   splitting "refund" from "return") would better match internal support
   tooling but makes the classification task harder to evaluate reliably at
   this dataset size, and the assignment explicitly asks for a "small set."

3. **Routing is rule-based, not another LLM call.** For decisions this
   consequential (money movement, account security, already-escalated
   customers), an auditable one-sentence rule beats an opaque model call --
   you can explain exactly why every escalation happened, which matters more
   here than marginal routing accuracy gains from a fancier classifier.

4. **All money/security/escalation intents (`refund_or_return`,
   `account_or_login_issue`, `billing_or_payment_issue`,
   `complaint_escalation`) are hard-coded to always escalate**, regardless of
   classifier confidence. The cost of a wrong auto-handled refund/security
   decision is asymmetric with the cost of an unnecessary escalation -- false
   auto-handles here are the failure mode that actually matters, so I
   optimized against that risk rather than against overall escalation
   accuracy.

5. **Retrieval uses TF-IDF + cosine similarity, not embeddings.** Given the
   time budget, I chose the cheaper option deliberately rather than testing
   both: real AmazonHelp tweets are short and formulaic (a handful of
   recurring templates -- "sorry, DM your order #", "refund issued in 3-5
   business days"), so exact/near-exact phrase overlap that TF-IDF captures
   well is doing most of the useful work here. I did not empirically compare
   this against an embedding-based retriever, so I can't rule out that
   embeddings would meaningfully improve grounding on paraphrased or
   non-templated complaints -- that's a clear next-week item (see "what I'd
   do next" in report.md).

6. **The drafter is explicitly instructed to ask for missing info rather
   than invent a resolution when no good grounding example exists**, instead
   of always producing a "confident-sounding" reply. This trades apparent
   reply quality for groundedness.

7. **LLM-judge rubric has 4 separate sub-scores (groundedness, helpfulness,
   tone, correctness) instead of one holistic score.** A single score lets a
   serious groundedness failure get averaged away by good tone; splitting
   them out keeps that failure mode visible.

8. **Confidence threshold for auto-handling is 0.6** (`CONFIDENCE_ESCALATE_THRESHOLD`
   in config.py). This was an arbitrary starting point, not tuned against
   the golden set's actual false-auto-handle rate -- with more time I'd sweep
   this threshold against the 154-example golden set and pick the value that
   best trades off unnecessary escalations against dangerous false
   auto-handles, rather than guessing 0.6 up front.

9. **Golden set sampling is stratified by intent, not proportional to
   real-world frequency** (see eval/labeling_guide.md) -- deliberately
   oversampling rare high-risk intents so the eval can actually say something
   about them, at the cost of the golden set not reflecting true traffic mix.

10. **Baseline "simple" classifier uses keyword matching, not a lightweight
    trained ML classifier (e.g. logistic regression on TF-IDF).** This was a
    time-budget call, not a claim that it's the more informative baseline --
    a trained classifier likely would have been a fairer "is the LLM earning
    its cost" comparison, since keyword matching is a much lower bar to
    clear than a model that's actually learned decision boundaries from
    data. I'd add this as a third baseline with more time.

11. **Multi-intent messages are labeled by their primary/most urgent intent**,
    with a note added explaining the secondary intent tangled in (e.g. a
    message asking to cancel a membership because account access is
    blocked is labeled `cancellation_request`, with a note flagging the
    access-issue sub-thread). This keeps the taxonomy clean-per-label while
    preserving the ambiguity as a documented edge case rather than hiding it.

12. **Decided NOT to fix the classification/taxonomy gaps found during
    evaluation** (the `other_spam_irrelevant` bucket wrongly absorbing
    resolved "thanks, all good!" messages, and `general_product_question`
    under-triggering on real product questions) despite having clear
    hypotheses for both. With the time remaining, writing an honest,
    evidence-backed failure analysis of a known problem was a better use of
    time than attempting a fix, re-running the full pipeline, and
    potentially introducing new unverified failure modes right before
    submission. See report.md's failure analysis for the full reasoning.

13. **Model choice: `openai/gpt-oss-20b` for classification/routing,
    `openai/gpt-oss-120b` for drafting/judging** (via Groq's free tier,
    chosen over a paid API for cost reasons). Classification and routing are
    comparatively simple, well-specified tasks where a smaller/faster model
    is sufficient and keeps latency and cost down across ~150+ calls per
    eval run; drafting a customer-facing reply and judging reply quality are
    higher-stakes, more nuanced tasks where the larger model's better
    language quality and judgment matter more, and there are far fewer such
    calls per run.

14. **Deliberately did not build:** multi-turn conversation handling (the
    agent scores/responds to a single inbound message, not a full thread --
    a different, harder problem involving conversation state); fine-tuning
    (used prompted LLM calls only, given the time and data-volume
    constraints); an embeddings-based retriever (see decision #5); and
    threshold auto-tuning (see decision #8). Each of these is a reasonable
    "next week" item rather than something I judged unnecessary for a
    working v1.

15. **Golden set restricted to English-language tweets.** Non-English
    messages were kept in the file but labeled `EXCLUDED_NON_ENGLISH` and
    skipped during evaluation, rather than deleted -- this preserves a
    visible record of what was excluded and why, rather than silently
    dropping them. A real deployment would need multilingual handling; this
    was a scope cut made for time, not a claim that non-English support
    traffic doesn't matter.
