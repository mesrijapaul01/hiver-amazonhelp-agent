"""
Classifies a customer message into one of the INTENTS defined in config.py.

Returns intent + a self-reported confidence (0-1) + a one-line rationale.
Self-reported LLM confidence is NOT calibrated probability -- it's a cheap
heuristic. We validate/calibrate it empirically in eval/ (see report.md,
"what's misleading about my headline number").
"""
from src.config import INTENTS, CLASSIFIER_MODEL
from src.llm_client import call_llm_json

SYSTEM = """You are an intent classifier for AmazonHelp customer support tweets.
Classify the customer's message into exactly one of the given intents.
Respond with ONLY a JSON object, no other text, no markdown fences:
{"intent": "<one of the intent keys>", "confidence": <float 0-1>, "rationale": "<one short sentence>"}
"""


def build_user_prompt(tweet: str) -> str:
    intent_list = "\n".join(f"- {k}: {v}" for k, v in INTENTS.items())
    return f"""Intents:
{intent_list}

Customer message:
\"\"\"{tweet}\"\"\"

Classify it."""


def classify(tweet: str) -> dict:
    result = call_llm_json(
        model=CLASSIFIER_MODEL,
        system=SYSTEM,
        user=build_user_prompt(tweet),
        max_tokens=200,
    )
    intent = result.get("intent", "other_spam_irrelevant")
    if intent not in INTENTS:
        # LLM hallucinated a label outside our taxonomy -- fall back safely
        intent = "other_spam_irrelevant"
        result["confidence"] = 0.0
        result["rationale"] = f"(unrecognized label '{result.get('intent')}', forced to fallback)"
    return {
        "intent": intent,
        "confidence": float(result.get("confidence", 0.5)),
        "rationale": result.get("rationale", ""),
    }
