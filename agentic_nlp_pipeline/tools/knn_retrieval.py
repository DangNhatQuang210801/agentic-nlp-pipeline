"""Retrieval tools for prompting agents."""

from pathlib import Path

from stanza.models.common.doc import Document, Sentence, TEXT, UPOS
from stanza.utils.conll import CoNLL

from .utils import (
    features,
    token_dicts_to_sentence,
    tool_error,
    tool_json,
    sentence_to_json,
)


class KNNRetrievalTool:
    """Brute-force train-treebank retrieval by FORM and UPOS n-gram overlap."""

    schema = {
        "type": "function",
        "function": {
            "name": "retrieve_similar_sentences",
            "description": (
                "Find similar annotated train-treebank sentences for a tokenized sentence. "
                "Similarity uses surface-form and optional UPOS n-gram overlap."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "description": "ISO-3 language code. Options: eng = English, mar = Marathi, nan = Taiwanese, nds = Low German / Low Saxon, vie = Vietnamese",
                    },
                    "tokens": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "form": {"type": "string"},
                                "upos": {"type": "string"},
                            },
                            "required": ["id", "form"],
                        },
                        "description": "Input sentence as a JSON-style list of token dictionaries.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of examples to return.",
                    },
                },
                "required": ["language", "tokens"],
            },
        },
    }

    def __init__(self, documents: dict[str, Document], max_n: int = 3):
        self.documents = documents
        self.max_n = max_n

    @classmethod
    def from_conllu_files(cls, train_files: dict[str, str | Path], max_n: int = 3):
        return cls(
            {
                language: CoNLL.conll2doc(input_file=str(path))
                for language, path in train_files.items()
            },
            max_n=max_n,
        )

    @classmethod
    def from_documents(cls, documents: dict[str, Document], max_n: int = 3):
        return cls(documents, max_n=max_n)

    def retrieve(self, language: str, sent: Sentence, k: int = 3):
        if language not in self.documents:
            raise ValueError(f"Unknown language: {language}")

        query = features(sent, self.max_n)
        scored = []
        for sentence in self.documents[language].sentences:
            sentence_features = features(sentence, self.max_n)
            union = query | sentence_features
            score = len(query & sentence_features) / len(union) if union else 0.0
            scored.append((score, sentence))

        return sorted(scored, key=lambda item: item[0], reverse=True)[:k]

    def search(
        self,
        language: str,
        tokens: list[dict],
        k: int = 3,
    ) -> str:
        try:
            sent = token_dicts_to_sentence(tokens)
            results = [
                sentence_to_json(score, sentence)
                for score, sentence in self.retrieve(language, sent, k)
            ]
            return tool_json(results)
        except ValueError as exc:
            return tool_error(str(exc))

    def as_agent_tool(self):
        return self.schema, self.search
