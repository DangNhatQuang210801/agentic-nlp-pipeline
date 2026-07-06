from typing import Protocol


class LanguageModel(Protocol):
    def complete(self): ...
