"""Small callable tools for prompting agents."""

from dataclasses import dataclass
import json
from pathlib import Path
from stanza.models.common.doc import Document, Sentence
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


@dataclass
class IndexedSentence:
    sent_id: str
    text: str
    forms: list[str]
    upos: list[str]
    tokens: list[dict[str, str]]
    features: set[tuple[str, tuple[str, ...]]]


def _ngrams(items: list[str], max_n: int) -> set[tuple[str, ...]]:
    grams = set()
    for n in range(1, max_n + 1):
        grams.update(tuple(items[i : i + n]) for i in range(len(items) - n + 1))
    return grams


def _features(forms: list[str], upos: list[str] | None, max_n: int):
    features = {("form", gram) for gram in _ngrams([f.casefold() for f in forms], max_n)}
    if upos:
        features.update(("upos", gram) for gram in _ngrams(upos, max_n))
    return features


def _token(word) -> dict[str, str]:
    return {
        "ID": str(word.id),
        "FORM": word.text or "_",
        "LEMMA": word.lemma or "_",
        "UPOS": word.upos or "_",
        "XPOS": word.xpos or "_",
        "FEATS": word.feats or "_",
        "HEAD": str(word.head),
        "DEPREL": word.deprel or "_",
    }


def _index_sentence(sent: Sentence, max_n: int) -> IndexedSentence:
    tokens = [_token(word) for word in sent.words]
    forms = [token["FORM"] for token in tokens]
    upos = [token["UPOS"] for token in tokens]
    return IndexedSentence(
        sent_id=getattr(sent, "sent_id", ""),
        text=getattr(sent, "text", None) or " ".join(forms),
        forms=forms,
        upos=upos,
        tokens=tokens,
        features=_features(forms, upos, max_n),
    )


def _index_doc(doc: Document, max_n: int) -> list[IndexedSentence]:
    return [_index_sentence(sent, max_n) for sent in doc.sentences]


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

    def __init__(self, indexes: dict[str, list[IndexedSentence]], max_n: int = 3):
        self.indexes = indexes
        self.max_n = max_n

    @classmethod
    def from_conllu_files(cls, train_files: dict[str, str | Path], max_n: int = 3):
        return cls(
            {
                language: _index_doc(CoNLL.conll2doc(input_file=str(path)), max_n)
                for language, path in train_files.items()
            },
            max_n=max_n,
        )

    @classmethod
    def from_documents(cls, documents: dict[str, Document], max_n: int = 3):
        return cls(
            {language: _index_doc(doc, max_n) for language, doc in documents.items()},
            max_n=max_n,
        )

    def retrieve(self, language: str, forms: list[str], upos: list[str] | None = None, k: int = 3):
        if language not in self.indexes:
            raise ValueError(f"Unknown language: {language}")
        if upos and len(upos) != len(forms):
            raise ValueError("upos must have the same length as forms")

        query = _features(forms, upos, self.max_n)
        scored = []
        for sentence in self.indexes[language]:
            union = query | sentence.features
            score = len(query & sentence.features) / len(union) if union else 0.0
            scored.append((score, sentence))

        results = []
        for score, sentence in sorted(scored, key=lambda item: item[0], reverse=True)[:k]:
            results.append(
                {
                    "score": round(score, 4),
                    "sent_id": sentence.sent_id,
                    "text": sentence.text,
                    "tokens": sentence.tokens,
                }
            )
        return results

    def search(self, language: str, forms: list[str], upos: list[str] | None = None, k: int = 3) -> str:
        try:
            return json.dumps(self.retrieve(language, forms, upos, k), ensure_ascii=False)
        except ValueError as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    def as_agent_tool(self):
        return self.schema, self.search
