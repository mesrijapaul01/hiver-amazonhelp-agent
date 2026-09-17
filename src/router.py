"""
Decides auto-handle vs escalate-to-human, with a stated reason.

Deliberately rule-based rather than another LLM call: for a decision this
consequential (money movement, account security, angry customers), an
auditable rule you can explain in one sentence beats an opaque model call.
See decision_log.md.
"""
from src.config import RISK_TIER, CONFIDENCE_ESCALATE_THRESHOLD

NO_GROUNDING_MIN_EXAMPLES = 1  # if we found zero grounding examples, don't auto-handle


def decide(intent: str, confidence: float, num_grounding_examples: int) -> dict:
    risk = RISK_TIER.get(intent, "high")  # unknown intent -> treat as high risk

    if risk == "high":
        return {
            "decision": "escalate",
            "reason": f"Intent '{intent}' is high-risk (money movement, security, or already-escalated customer) -- always routed to a human regardless of classifier confidence.",
        }

    if confidence < CONFIDENCE_ESCALATE_THRESHOLD:
        return {
            "decision": "escalate",
            "reason": f"Classifier confidence ({confidence:.2f}) is below the {CONFIDENCE_ESCALATE_THRESHOLD} threshold -- too uncertain to auto-handle.",
        }

    if num_grounding_examples < NO_GROUNDING_MIN_EXAMPLES:
        return {
            "decision": "escalate",
            "reason": "No sufficiently similar historical resolution found to ground a reply -- drafting blind is riskier than handing to a human.",
        }

    return {
        "decision": "auto_handle",
        "reason": f"Intent '{intent}' is {risk}-risk, classifier confidence ({confidence:.2f}) meets threshold, and a grounded historical resolution exists.",
    }
