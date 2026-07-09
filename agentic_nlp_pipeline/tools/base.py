from typing import Callable, Protocol


class AgentTool(Protocol):
    """Protocol for all agent tools.

    Attributes:
        schema: The tool's schema.
    """

    schema: dict

    def as_agent_tool(self) -> tuple[dict, Callable]: ...
