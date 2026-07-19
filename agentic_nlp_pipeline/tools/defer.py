"""Defer tool: postpone a head decision instead of guessing."""

from typing import Callable


class DeferTool:
    """Let the model postpone unresolved nodes instead of hallucinating a head."""

    schema = {
        "type": "function",
        "function": {
            "name": "defer",
            "description": (
                "Postpone a decision for one or more token IDs instead of guessing "
                "immediately. Use for unfamiliar morphology (OOV) or genuinely "
                "unclear attachment. You must still resolve every deferred node "
                "before your final answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Token IDs to postpone a decision for.",
                    },
                    "reason": {
                        "type": "string",
                        "enum": ["oov", "ambiguous_attachment", "insufficient_evidence"],
                    },
                },
                "required": ["ids", "reason"],
            },
        },
    }

    @classmethod
    def defer(cls, ids: list[int], reason: str) -> str:
        """Acknowledge a deferred decision; resolves nothing itself.

        Args:
            ids: Token IDs being postponed.
            reason: Why they're being postponed.

        Returns:
            A confirmation message fed back to the model as the tool result.
        """
        if not ids:
            return "No IDs provided -- nothing was deferred."
        return (
            f"Deferred {ids} (reason: {reason}). Continue to the next word; "
            "you must still assign each of these a head before your final answer."
        )

    @classmethod
    def as_agent_tool(cls) -> tuple[dict, Callable]:
        return cls.schema, cls.defer