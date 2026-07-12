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
