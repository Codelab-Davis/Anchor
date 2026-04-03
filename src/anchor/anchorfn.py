from __future__ import annotations

from abc import ABC, abstractmethod


class AnchorFn(ABC):
    """Base class for model callables passed to the simple interface.

    Subclass this and implement __call__ to wrap any LLM.

    Example::

        class MyAI(AnchorFn):
            def __call__(self, messages: list[dict]) -> str:
                return client.chat.completions.create(
                    model="gpt-4.1-mini", messages=messages
                ).choices[0].message.content
    """

    @abstractmethod
    def __call__(self, messages: list[dict]) -> str:
        """Call the model.

        Args:
            messages: List of OpenAI-format message dicts
                      (e.g. ``[{"role": "user", "content": "..."}]``).

        Returns:
            The model's response as a plain string.
        """
