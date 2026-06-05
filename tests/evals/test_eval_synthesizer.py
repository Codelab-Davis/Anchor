from __future__ import annotations

import pytest

from anchor.synthesizer import Synthesizer


# Three chunks with overlapping entities (Elena Marsh, NovaTech) and complementary
# facts (founding/HQ, product details, founder background).
CHUNKS = [
    {
        "source": "company_overview",
        "content": (
            "Elena Marsh founded NovaTech in 2018. "
            "The company is headquartered in Austin, Texas."
        ),
    },
    {
        "source": "product_brief",
        "content": (
            "NovaTech's flagship product is Prism, a platform that processes "
            "satellite imagery for agricultural monitoring. Prism launched in 2020."
        ),
    },
    {
        "source": "founder_bio",
        "content": (
            "Elena Marsh holds a PhD in remote sensing from Stanford University "
            "and has over 15 years of experience in the geospatial industry."
        ),
    },
]

# Two-question context so the (current) marker lands on the second question.
QUESTIONS = [
    "What companies use satellite imagery?",
    "Who founded NovaTech and what does their main product do?",
]

_REFUSAL_PHRASES = ("i cannot", "i can't", "i'm unable", "no information")

# Entities that appear nowhere in the chunks above.
_ABSENT_ENTITIES = ("Tesla", "Elon Musk", "Python", "OpenAI", "Microsoft")


@pytest.mark.eval
@pytest.mark.parametrize("run_number", range(5))
def test_eval_synthesizer_synthesize(ai_fn, run_number):
    synthesizer = Synthesizer(model_fn=ai_fn)
    output = synthesizer.synthesize(CHUNKS, QUESTIONS)

    # Not empty or a refusal.
    assert output and output.strip(), f"[run {run_number}] output was empty"
    assert not any(p in output.lower() for p in _REFUSAL_PHRASES), (
        f"[run {run_number}] output looks like a refusal: {output!r}"
    )

    # Key facts spanning all three chunks must appear in the synthesis.
    for fact in ("Elena Marsh", "NovaTech", "Prism", "satellite"):
        assert fact.lower() in output.lower(), (
            f"[run {run_number}] missing expected fact {fact!r}: {output!r}"
        )

    # Current question asks about the founder — must be addressed.
    assert "elena" in output.lower() or "founder" in output.lower(), (
        f"[run {run_number}] founder topic not addressed: {output!r}"
    )

    # Current question asks about the main product — must be addressed.
    assert "prism" in output.lower() or "satellite" in output.lower(), (
        f"[run {run_number}] product topic not addressed: {output!r}"
    )

    # Model must not hallucinate entities absent from the input chunks.
    for entity in _ABSENT_ENTITIES:
        assert entity.lower() not in output.lower(), (
            f"[run {run_number}] hallucinated entity {entity!r}: {output!r}"
        )
