import json
from typing import Any, Iterable

from stanza.models.common.doc import (
    DEPREL,
    FEATS,
    HEAD,
    ID,
    LEMMA,
    Sentence,
    TEXT,
    UPOS,
    XPOS,
)


def sentence_to_token_dicts(
    sent: Sentence,
    fields: Iterable[str] = ("id", "form", "upos"),
) -> list[dict[str, Any]]:
    """Convert a Stanza sentence to the JSON token format shown to models."""
    tokens = []
    for word in sent.words:
        token = {}
        if "id" in fields:
            token["id"] = word.id
        if "form" in fields:
            token["form"] = word.text or "_"
        if "lemma" in fields:
            token["lemma"] = word.lemma or "_"
        if "upos" in fields:
            token["upos"] = word.upos or "_"
        if "xpos" in fields:
            token["xpos"] = word.xpos or "_"
        if "feats" in fields:
            token["feats"] = word.feats or "_"
        if "head" in fields:
            token["head"] = word.head
        if "deprel" in fields:
            token["deprel"] = word.deprel or "_"
        tokens.append(token)
    return tokens


def token_dicts_to_sentence(tokens: list[dict[str, Any]]) -> Sentence:
    """Build a Stanza sentence from parsed model/tool token dictionaries."""
    if isinstance(tokens, str):
        tokens = json.loads(tokens)
    if not isinstance(tokens, list) or not all(isinstance(t, dict) for t in tokens):
        raise TypeError(f"Expected list of token dicts, got: {tokens!r}")
    return Sentence(
        [
            {
                ID: int(token.get("id", i + 1)),
                TEXT: str(token.get("form", token.get("text", "_"))),
                LEMMA: token.get("lemma"),
                UPOS: token.get("upos", "_"),
                XPOS: token.get("xpos"),
                FEATS: token.get("feats"),
                HEAD: token.get("head"),
                DEPREL: token.get("deprel"),
            }
            for i, token in enumerate(tokens)
        ]
    )


def tool_json(payload: Any) -> str:
    """Return model-readable JSON from a tool."""
    return json.dumps(payload, ensure_ascii=False)


def tool_error(message: str) -> str:
    return tool_json({"error": message})


def ngrams(items: list[str], max_n: int) -> set[tuple[str, ...]]:
    grams = set()
    for n in range(1, max_n + 1):
        grams.update(tuple(items[i : i + n]) for i in range(len(items) - n + 1))
    return grams


def features(sent: Sentence, max_n: int) -> set[tuple[str, tuple[str, ...]]]:
    features = {
        ("FORM", gram)
        for gram in ngrams(
            [(getattr(word, TEXT) or "_").casefold() for word in sent.words],
            max_n,
        )
    }
    features.update(
        ("UPOS", gram)
        for gram in ngrams(
            [getattr(word, UPOS) or "_" for word in sent.words],
            max_n,
        )
    )
    return features


def sentence_to_json(score: float, sent: Sentence) -> dict:
    return {
        "score": round(score, 4),
        "sent_id": getattr(sent, "sent_id", ""),
        "text": getattr(sent, "text", None)
        or " ".join(getattr(word, TEXT) or "_" for word in sent.words),
        "tokens": sentence_to_token_dicts(
            sent,
            fields=("id", "form", "lemma", "upos", "xpos", "feats", "head", "deprel"),
        ),
    }
