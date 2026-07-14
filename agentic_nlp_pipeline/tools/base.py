from collections import Counter
from pathlib import Path
from typing import Any, Callable, Protocol

from stanza.models.common.doc import Document
from stanza.utils.conll import CoNLL

from .utils import tool_error, tool_json


class AgentTool(Protocol):
    """Protocol for all agent tools.

    The harness parses raw model tool calls into Python arguments before
    calling a tool. Tools return model-readable strings, usually JSON.
    """

    schema: dict[str, Any]

    def as_agent_tool(self) -> tuple[dict[str, Any], Callable[..., str]]: ...


class MorphologyLookupTool:
    """Look up token analyses observed in multilingual CoNLL-U training data."""

    schema = {
        "type": "function",
        "function": {
            "name": "lookup_morphology",
            "description": "Return morphological analyses seen in a training treebank.",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {"type": "string"},
                    "tokens": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "form": {"type": "string"},
                            },
                            "required": ["id", "form"],
                        },
                    },
                },
                "required": ["language", "tokens"],
            },
        },
    }

    def __init__(self, documents: dict[str, Document]):
        self.index = {}
        for language, document in documents.items():
            language_index = {}
            for word in document.iter_words():
                fields = language_index.setdefault(
                    (word.text or "_").casefold(),
                    {"lemma": Counter(), "upos": Counter(), "feats": Counter()},
                )
                fields["lemma"][word.lemma or "_"] += 1
                fields["upos"][word.upos or "_"] += 1
                fields["feats"][word.feats or "_"] += 1
            self.index[language] = language_index

    @classmethod
    def from_conllu_files(cls, train_files: dict[str, str | Path]):
        return cls(
            {
                language: CoNLL.conll2doc(input_file=str(path))
                for language, path in train_files.items()
            }
        )

    def lookup(self, language: str, tokens: list[dict]) -> str:
        if language not in self.index:
            return tool_error(f"Unknown language: {language}")
        if any("id" not in token or "form" not in token for token in tokens):
            return tool_error("Each token needs an id and form.")

        results = []
        for token in tokens:
            form = str(token["form"])
            fields = self.index[language].get(form.casefold(), {})
            result = {"id": token["id"], "form": form}
            for field in ("lemma", "upos", "feats"):
                result[f"{field}_candidates"] = [
                    {"value": value, "count": count}
                    for value, count in fields.get(field, Counter()).most_common(5)
                ]
            results.append(result)
        return tool_json(results)

    def as_agent_tool(self):
        return self.schema, self.lookup
