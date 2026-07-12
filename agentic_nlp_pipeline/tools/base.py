from typing import Any, Callable, Protocol


class AgentTool(Protocol):
    """Protocol for all agent tools.

    The harness parses raw model tool calls into Python arguments before
    calling a tool. Tools return model-readable strings, usually JSON.
    """

    schema: dict[str, Any]

    def as_agent_tool(self) -> tuple[dict[str, Any], Callable[..., str]]: ...
