"""
Replaces route_question_to_lens() from chart_engine.py with a real
language-understanding classifier instead of keyword matching.

Requires ANTHROPIC_API_KEY set in the environment. In production this
lives in the same backend that calls chart_engine—the Next.js API
route would call this (or a Python service running it) before touching
the transit scanner at all.
"""

import json
import os
from anthropic import Anthropic

VALID_LENSES = ["timing", "money", "career", "relationships", "location"]

CLASSIFY_PROMPT = """You are routing a user's astrology question to the correct
calculation engine. Read the question and respond with ONLY a JSON object,
no other text, in this exact shape:

{{"lens": "<one of: timing, money, career, relationships, location>",
  "reasoning": "<one short sentence>",
  "date_range_hint": "<any timeframe the user mentioned, or null>",
  "location_mentioned": "<any place name the user mentioned, or null>"}}

Lens definitions:
- timing: general good/bad day questions with no clear money/career/relationship/location angle
- money: income, investing, spending, financial decisions
- career: work, business, launches, promotions, professional moves
- relationships: dating, partnership, romantic timing
- location: whether a specific place or move would suit them (this routes to
  astrocartography, NOT the transit scanner—flag it even if the question
  also has a timing element)

User's question: "{question}"
"""


def route_question_with_llm(question_text, client=None):
    """
    Returns a dict: {lens, reasoning, date_range_hint, location_mentioned}.
    Falls back to lens="timing" with an error note if the API call fails
    or returns something unparseable—routing should never hard-crash
    the user's question, it should degrade to the safest general default.
    """
    if client is None:
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",  # fast/cheap—this is a simple routing task
            max_tokens=200,
            messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(question=question_text)}],
        )
        raw = response.content[0].text.strip()
        parsed = json.loads(raw)
        if parsed.get("lens") not in VALID_LENSES:
            raise ValueError(f"Model returned an unrecognized lens: {parsed.get('lens')}")
        return parsed
    except Exception as e:
        return {
            "lens": "timing",
            "reasoning": f"Routing failed, defaulted to general timing: {e}",
            "date_range_hint": None,
            "location_mentioned": None,
        }


if __name__ == "__main__":
    test_questions = [
        "When's a good day to launch my new business?",
        "Should I invest more this month, or is now a bad time financially?",
        "Is this a good week to go on a date?",
        "Would moving to Austin be good for me astrologically?",
        "My landlord wants an answer by Friday, is that a fine day to sign the lease?",
    ]
    for q in test_questions:
        result = route_question_with_llm(q)
        print(f'"{q}"\n  -> {result}\n')
