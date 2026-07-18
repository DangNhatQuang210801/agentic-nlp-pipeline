"""Dependency tree validation exposed through the shared tool protocol."""

from typing import Callable

from .utils import token_dicts_to_sentence
from ..validation import validate_sentence


class TreeValidationTool:
    """Validate token heads and return a concise structural assessment."""

    schema = {
        "type": "function",
        "function": {
            "name": "validate_dependency_tree",
            "description": (
                "Check whether a dependency tree adheres to formal constraints "
                "before submitting a final answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sent": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "head": {"type": "integer"},
                            },
                            "required": ["id", "head"],
                        },
                        "description": "The dependency tree as an array of token objects.",
                    },
                },
                "required": ["sent"],
            },
        },
    }

    @classmethod
    def validate_dependency_tree(cls, sent: list[dict[str, str | int]]) -> str:
        """Check if a Sentence has a valid dependency tree.

        Args:
            sent: Some Stanza Sentence.

        Returns:
            valid: Whether or not the dependency tree is valid.
            msg: A detailed success or error message.
        """
        _, msg = validate_sentence(token_dicts_to_sentence(sent))
        return msg

    @classmethod
    def as_agent_tool(cls) -> tuple[dict, Callable]:
        """Return the schema and validation callable."""
        return cls.schema, cls.validate_dependency_tree
