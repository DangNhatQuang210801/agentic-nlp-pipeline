"""Small callable tools for prompting agents."""

import json
from pathlib import Path
from stanza.models.common.doc import Document, ID, Sentence, TEXT, UPOS
from stanza.utils.conll import CoNLL


TOOL_TEMPLATE = {
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "...",
        "parameters": {
            "type": "object",
            "properties": {
                "param": {
                    "type": "string",
                    "description": "...",
                }
            },
            "required": ["param"],
        },
    },
}


def _ngrams(items: list[str], max_n: int) -> set[tuple[str, ...]]:
    grams = set()
    for n in range(1, max_n + 1):
        grams.update(tuple(items[i : i + n]) for i in range(len(items) - n + 1))
    return grams


def _features(sent: Sentence, max_n: int) -> set[tuple[str, tuple[str, ...]]]:
    features = {
        ("FORM", gram)
        for gram in _ngrams(
            [(getattr(word, TEXT) or "_").casefold() for word in sent.words],
            max_n,
        )
    }
    features.update(
        ("UPOS", gram)
        for gram in _ngrams(
            [getattr(word, UPOS) or "_" for word in sent.words],
            max_n,
        )
    )
    return features


def _sentence_to_json(score: float, sent: Sentence) -> dict:
    return {
        "score": round(score, 4),
        "sent_id": getattr(sent, "sent_id", ""),
        "text": getattr(sent, "text", None)
        or " ".join(getattr(word, TEXT) or "_" for word in sent.words),
        "tokens": sent.to_dict(),
    }




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
                    "language": {"type": "string", "description": "Language key, e.g. vietnamese."},
                    "forms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Token FORM values in sentence order.",
                    },
                    "upos": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional UPOS tags in sentence order.",
                    },
                    "k": {"type": "integer", "description": "Number of examples to return."},
                },
                "required": ["language", "forms"],
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

        query = _features(sent, self.max_n)
        scored = []
        for sentence in self.documents[language].sentences:
            sentence_features = _features(sentence, self.max_n)
            union = query | sentence_features
            score = len(query & sentence_features) / len(union) if union else 0.0
            scored.append((score, sentence))

        return sorted(scored, key=lambda item: item[0], reverse=True)[:k]

    def search(self, language: str, forms: list[str], upos: list[str] | None = None, k: int = 3) -> str:
        try:
            if upos and len(upos) != len(forms):
                raise ValueError("upos must have the same length as forms")
            sent = Sentence(
                [
                    {ID: i + 1, TEXT: form, UPOS: upos[i] if upos else "_"}
                    for i, form in enumerate(forms)
                ]
            )
            results = [
                _sentence_to_json(score, sentence)
                for score, sentence in self.retrieve(language, sent, k)
            ]
            return json.dumps(results, ensure_ascii=False)
        except ValueError as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    def as_agent_tool(self):
        return self.schema, self.search
