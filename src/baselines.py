"""
Two baselines the report compares the full agent against:

1. TRIVIAL baseline: always predicts the majority-class intent, always
   escalates every message, never drafts a reply. This is the floor --
   if the agent doesn't beat this by a wide margin something is broken.

2. SIMPLE baseline: keyword-matching intent classifier (no LLM) + a fixed
   template reply per intent + the same rule-based router. Tests whether
   the LLM classifier/drafter is actually earning its cost over cheap
   string matching.
"""
from src.config import INTENTS, RISK_TIER
from src.router import decide

MAJORITY_INTENT = "refund_or_return"  # most common label in the real golden set (154-row eval set)

KEYWORD_RULES = {
    "refund_or_return": ["refund", "return", "money back"],
    "damaged_or_wrong_item": ["damaged", "broken", "wrong item", "defective", "shattered", "cracked"],
    "delivery_delay_or_lost": ["late", "lost", "never got it", "not received", "delayed"],
    "cancellation_request": ["cancel"],
    "account_or_login_issue": ["password", "login", "locked out", "account suspended", "2fa"],
    "billing_or_payment_issue": ["charged twice", "double charged", "promo code", "gift card", "billing"],
    "complaint_escalation": ["manager", "human", "unacceptable", "chargeback", "consumer protection", "third time"],
    "general_product_question": ["does this come with", "is this compatible", "what's the difference"],
    "order_status_tracking": ["where is my", "tracking", "status", "arrive"],
}

TEMPLATE_REPLIES = {
    "order_status_tracking": "Thanks for reaching out! Could you share your order number via DM so we can check the latest status for you?",
    "delivery_delay_or_lost": "Sorry for the trouble! Please DM your order number and we'll look into this right away.",
    "damaged_or_wrong_item": "So sorry to hear that! Please DM your order number and photos if possible, we'll get this fixed.",
    "refund_or_return": "Thanks for letting us know! Please DM your order number so we can check the refund/return status.",
    "cancellation_request": "We can help with that! Please DM your order number and we'll process the cancellation.",
    "account_or_login_issue": "Sorry for the trouble accessing your account. Please DM us (not publicly) so we can verify and help securely.",
    "billing_or_payment_issue": "Sorry about that! Please DM your order number and we'll look into the charge.",
    "general_product_question": "Great question! Please check the product listing's details section, or DM us the item and we'll help clarify.",
    "complaint_escalation": "We're sorry for the frustration. Please DM us and we'll escalate this to a specialist right away.",
    "other_spam_irrelevant": "",
}


def trivial_predict(tweet: str) -> dict:
    routing = {"decision": "escalate", "reason": "Trivial baseline always escalates."}
    return {
        "predicted_intent": MAJORITY_INTENT,
        "draft_reply": "",
        "routing_decision": routing["decision"],
        "routing_reason": routing["reason"],
    }


def simple_keyword_predict(tweet: str) -> dict:
    t = tweet.lower()
    matched_intent = None
    for intent, keywords in KEYWORD_RULES.items():
        if any(kw in t for kw in keywords):
            matched_intent = intent
            break
    if matched_intent is None:
        matched_intent = "other_spam_irrelevant"

    # crude confidence: 1.0 if a keyword matched, 0.3 if we fell back
    confidence = 1.0 if matched_intent != "other_spam_irrelevant" else 0.3
    routing = decide(intent=matched_intent, confidence=confidence, num_grounding_examples=1)

    return {
        "predicted_intent": matched_intent,
        "draft_reply": TEMPLATE_REPLIES.get(matched_intent, ""),
        "routing_decision": routing["decision"],
        "routing_reason": routing["reason"],
    }
