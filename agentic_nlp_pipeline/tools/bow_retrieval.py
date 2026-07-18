"""Bag of words retrieval for annotated treebank sentences."""

from collections import Counter
from pathlib import Path

from stanza.models.common.doc import Document, Sentence
from stanza.utils.conll import CoNLL

from .utils import token_dicts_to_sentence, tool_error, tool_json, sentence_to_json


def _score(query: Sentence, candidate: Sentence) -> float:
    """Combine word overlap with sentence length similarity."""
    query_words = Counter((word.text or "_").casefold() for word in query.words)
    candidate_words = Counter((word.text or "_").casefold() for word in candidate.words)

    union_size = sum((query_words | candidate_words).values())
    overlap = (
        sum((query_words & candidate_words).values()) / union_size
        if union_size
        else 1.0
    )

    max_length = max(len(query.words), len(candidate.words))
    length_similarity = (
        min(len(query.words), len(candidate.words)) / max_length if max_length else 1.0
    )
    return (overlap + length_similarity) / 2


class BagOfWordsRetrievalTool:
    """Retrieve annotated sentences by word overlap and sentence length.

    Documents are stored by language. ``search`` accepts token dictionaries
    and returns JSON examples with complete CoNLL U annotations.
    """

    schema = {
        "type": "function",
        "function": {
            "name": "retrieve_similar_sentences_bow",
            "description": (
                "Find annotated sentences with similar length and content. The content similarity is estimated by calculating a bag-of-words similarity score."
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
                            },
                            "required": ["id", "form"],
                        },
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

    def __init__(self, documents: dict[str, Document]):
        """Store training documents by language."""
        self.documents = documents

    @classmethod
    def from_conllu_files(cls, train_files: dict[str, str | Path]):
        """Load one training file per language."""
        return cls(
            {
                language: CoNLL.conll2doc(input_file=str(path))
                for language, path in train_files.items()
            }
        )

    @classmethod
    def from_documents(cls, documents: dict[str, Document]):
        """Build the tool from existing Stanza documents."""
        return cls(documents)

    def retrieve(self, language: str, sentence: Sentence, k: int = 3):
        """Return the highest scoring annotated sentences."""
        if language not in self.documents:
            raise ValueError(f"Unknown language: {language}")

        scored = [
            (_score(sentence, candidate), candidate)
            for candidate in self.documents[language].sentences
        ]
        return sorted(scored, key=lambda item: item[0], reverse=True)[:k]

    def search(self, language: str, tokens: list[dict], k: int = 3) -> str:
        """Return retrieved sentences as JSON."""
        try:
            sentence = token_dicts_to_sentence(tokens)
            results = [
                sentence_to_json(score, candidate)
                for score, candidate in self.retrieve(language, sentence, k)
            ]
            return tool_json(results)
        except ValueError as exc:
            return tool_error(str(exc))

    def as_agent_tool(self):
        """Return the schema and search callable."""
        return self.schema, self.search
