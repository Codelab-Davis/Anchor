from __future__ import annotations

from anchor import Anchor


def _dummy_ai(_messages: list[dict]) -> str:
    return "DONE"


def test_anchor_config_reflects_max_remembers() -> None:
    anchor = Anchor(ai_fn=_dummy_ai, light_ai_fn=_dummy_ai)
    assert anchor.config().max_remembers == anchor.MAX_REMEMBERS
