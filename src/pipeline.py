"""
Ties classify -> retrieve -> draft -> route together for a single incoming
customer message. This is the function the eval harness and run_pipeline.py
both call, so there's exactly one code path for "what does the agent do".
"""
from src.classify_intent import classify
from src.retrieval import HistoricalRetriever
from src.draft_reply import draft
from src.router import decide

_retriever = None


def get_retriever(historical_csv: str = "data/historical_threads.csv") -> HistoricalRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HistoricalRetriever(historical_csv)
    return _retriever


def process_message(tweet: str, historical_csv: str = "data/historical_threads.csv") -> dict:
    retriever = get_retriever(historical_csv)

    intent_result = classify(tweet)
    examples = retriever.retrieve(tweet, intent=intent_result["intent"], k=3)
    reply = draft(tweet, intent_result["intent"], examples)
    routing = decide(
        intent=intent_result["intent"],
        confidence=intent_result["confidence"],
        num_grounding_examples=len(examples),
    )

    return {
        "customer_tweet": tweet,
        "predicted_intent": intent_result["intent"],
        "intent_confidence": intent_result["confidence"],
        "intent_rationale": intent_result["rationale"],
        "num_grounding_examples": len(examples),
        "grounding_example_similarities": [round(e["similarity"], 3) for e in examples],
        "draft_reply": reply,
        "routing_decision": routing["decision"],
        "routing_reason": routing["reason"],
    }
