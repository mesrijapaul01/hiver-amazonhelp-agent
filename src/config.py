"""
Central config: intent taxonomy, routing risk tiers, model names.

Intent taxonomy was derived by skimming a sample of AmazonHelp threads and
clustering the recurring issue types by hand (see decision_log.md, decision #1).
Keep this list small and mutually-exclusive-ish; that's the point of the exercise.
"""

INTENTS = {
    "order_status_tracking": "Customer wants to know where their order/package is, no complaint of delay yet.",
    "delivery_delay_or_lost": "Order is late, marked delivered but not received, or lost in transit.",
    "damaged_or_wrong_item": "Item arrived damaged, defective, or the wrong item was sent.",
    "refund_or_return": "Customer wants a refund, wants to return an item, or is asking about refund status.",
    "cancellation_request": "Customer wants to cancel an order or subscription before it ships/renews.",
    "account_or_login_issue": "Login problems, account locked, password reset, unauthorized access.",
    "billing_or_payment_issue": "Wrong charge, duplicate charge, payment failed, promo/gift card not applied.",
    "general_product_question": "Pre-purchase or how-to question about a product, not an issue with an existing order.",
    "complaint_escalation": "Customer is angry, has complained before, is threatening to leave/sue/report, or explicitly asks for a human/manager.",
    "other_spam_irrelevant": "Not a genuine support request (spam, unrelated chatter, ambiguous/unparseable).",
}

# Risk tiers drive the auto-handle vs escalate default *before* confidence is even considered.
# HIGH risk = money movement, security, or an already-escalated/angry customer -> never auto-handle.
RISK_TIER = {
    "order_status_tracking": "low",
    "delivery_delay_or_lost": "medium",
    "damaged_or_wrong_item": "medium",
    "refund_or_return": "high",
    "cancellation_request": "medium",
    "account_or_login_issue": "high",
    "billing_or_payment_issue": "high",
    "general_product_question": "low",
    "complaint_escalation": "high",
    "other_spam_irrelevant": "low",
}

# Confidence threshold below which we escalate regardless of risk tier.
CONFIDENCE_ESCALATE_THRESHOLD = 0.6

# Using Groq's free tier (OpenAI-compatible endpoints, open models).
# llama-3.3-70b-versatile was deprecated on Groq's free/developer tier in 2026 --
# these are Groq's current recommended replacements. If Groq deprecates these too,
# check https://console.groq.com/docs/deprecations and swap the string here.
CLASSIFIER_MODEL = "openai/gpt-oss-20b"   # fast + cheap, fine for classification/routing
DRAFTER_MODEL = "openai/gpt-oss-120b"      # stronger, used for reply drafting
JUDGE_MODEL = "openai/gpt-oss-120b"
