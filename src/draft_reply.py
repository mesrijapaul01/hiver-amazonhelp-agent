"""
Drafts a reply grounded in how this brand has historically resolved similar
issues (retrieved via src/retrieval.py). Explicitly instructed not to invent
policy/promises the retrieved examples don't support.
"""
from src.config import DRAFTER_MODEL
from src.llm_client import call_llm

SYSTEM = """You are drafting an AmazonHelp customer support reply on behalf of a human agent
who will review it before sending. Match the brand's real tone from the examples:
warm, apologetic where appropriate, concise, and always offering a concrete next step.

Rules:
- Ground your reply in the pattern of the example resolutions below -- do not invent
  policies, refund amounts, or timelines that aren't supported by the examples or the
  customer's message.
- If the examples don't clearly cover this situation, draft a reply that acknowledges
  the issue and asks for the specific info needed (order #, account email, etc.) rather
  than guessing at a resolution.
- Keep it under ~280 characters where possible, matching real tweet-reply length.
- Sign off in the same style as the examples (a first name).
- Output ONLY the reply text, nothing else."""


def build_user_prompt(tweet: str, intent: str, examples: list[dict]) -> str:
    if examples:
        ex_text = "\n\n".join(
            f"Similar past case:\nCustomer: {e['customer_tweet']}\nAgent resolution: {e['historical_agent_reply']}"
            for e in examples
        )
    else:
        ex_text = "(No closely similar past resolution found -- ask for the specific details needed rather than guessing.)"

    return f"""Predicted intent: {intent}

{ex_text}

Now draft a reply to this new customer message:
\"\"\"{tweet}\"\"\""""


def draft(tweet: str, intent: str, examples: list[dict]) -> str:
    return call_llm(
        model=DRAFTER_MODEL,
        system=SYSTEM,
        user=build_user_prompt(tweet, intent, examples),
        max_tokens=300,
    ).strip()
