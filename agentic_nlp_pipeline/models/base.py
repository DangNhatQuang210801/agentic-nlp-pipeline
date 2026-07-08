from typing import Optional, Protocol


class LanguageModel(Protocol):
    """Base protocol for all language models"""

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        tools: Optional[list[dict]] = None,
    ) -> str: ...
