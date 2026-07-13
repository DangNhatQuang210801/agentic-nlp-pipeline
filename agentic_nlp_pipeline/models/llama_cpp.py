import json
from typing import cast
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam


class LlamaCppModel:
    """Wrapper for locally served language models."""

    def __init__(
        self,
        model_name: str = "default",
        base_url: str = "http://localhost:8080/v1",
    ):
        base_url = "http://localhost:8080/v1"
        print(f"Connecting to local server at `{base_url}`")

        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.model_name = model_name

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        tools: list[dict] | None = None,
    ) -> str:
        """Generate a model response.

        Args:
            messages: A list of messages, each with a `role` and `content`.
            max_new_tokens: The maximum number of tokens to be generated.
            tools: Optional list of tool schemas in OpenAI function-calling
                format.

        Returns:
            The model response as a string. If the model made tool calls,
            they are serialized into the same pseudo-XML format our
            existing _parse_tool_calls expects.
        """
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=cast(list[ChatCompletionMessageParam], messages),
            tools=cast(list[ChatCompletionToolParam], tools) if tools else [],
            max_tokens=max_new_tokens,
            stream=True,
        )

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_call_chunks: dict[int, dict] = {}

        for chunk in stream:
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)

            if reasoning:
                reasoning_parts.append(reasoning)
                print(reasoning, end="", flush=True)
            if delta.content:
                content_parts.append(delta.content)
                print(delta.content, end="", flush=True)

            for tc_delta in delta.tool_calls or []:
                entry = tool_call_chunks.setdefault(
                    tc_delta.index, {"name": "", "arguments": ""}
                )
                if tc_delta.function and tc_delta.function.name:
                    entry["name"] += tc_delta.function.name
                if tc_delta.function and tc_delta.function.arguments:
                    entry["arguments"] += tc_delta.function.arguments
        print()

        # It would probably have been easier to work with the
        # outputs directly, but stitching them back together inot
        # one response lets us use our existing parsing functions.
        parts = []
        if reasoning_parts:
            parts.append(f"<think>\n{''.join(reasoning_parts)}\n</think>")
        if content_parts:
            parts.append("".join(content_parts).strip())
        for entry in tool_call_chunks.values():
            args = json.loads(entry["arguments"])
            params = "\n".join(
                f"<parameter={key}>\n{value}\n</parameter>"
                for key, value in args.items()
            )
            parts.append(
                f"<tool_call>\n<function={entry['name']}>\n{params}\n</function>\n</tool_call>"
            )
        return "\n".join(parts).strip()


# Little test
if __name__ == "__main__":
    model = LlamaCppModel()

    system_prompt = "You are a precise, concise linguistics expert."
    turn = "Who are you?"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": turn},
    ]
    reply = model.generate(messages)

    print(f"\n👨‍🦲 User: {turn}")
    print(f"🤖 Bot : {reply}")
