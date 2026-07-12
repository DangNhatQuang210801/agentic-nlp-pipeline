from datetime import datetime
import json
from pathlib import Path
from typing import Callable

import regex as re
from stanza.models.common.doc import Sentence

from agentic_nlp_pipeline import LanguageModel


# Some of this code is losely based on the exercise to Lecture 07.
class DepParseAgent:
    TOOLS = []
    TOOL_REGISTRY = {}

    def __init__(
        self,
        model: LanguageModel,
        system_prompt: str,
        max_new_tokens: int,
        max_iters_per_call: int,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.max_iters_per_call = max_iters_per_call

    def register_tool(self, tool_schema: dict, method: Callable):
        """Register an agent tool.

        Args:
            tool: The tool's defining schema.
            method: The function associated to the tool.
        """
        self.TOOLS.append(tool_schema)
        self.TOOL_REGISTRY[tool_schema["function"]["name"]] = method

    def run(self, sent: Sentence) -> Sentence:
        """Parse the dependencies of a sentence.

        Args:
            sent: The sentence to be parsed.

        Returns:
            A new sentence with HEAD attributes set by the agent.
        """
        sent_text = sent.text
        sent_words = [{"id": w.id, "form": w.text} for w in sent.words]

        content = f"Sentence: {sent_text}\nStructure:\n{str(sent_words)}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]

        for msg in messages:
            self._print_msg(msg)

        for _ in range(self.max_iters_per_call):
            messages = self._run_iteration(messages)
            if messages[-1]["role"] != "tool":
                break
            # TODO: Break only if output is satisfactory

        self._log_run(messages)

        tree = self._parse_dep_tree(messages[-1]["content"])
        for word, edge in zip(sent.words, tree):
            word.head = edge["head"]
        return sent

    def _run_iteration(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Run one iteration of obtaining a model response and making
        executing tool calls.

        Args:
            messages: The list of messages as context.

        Returns:
            A new list of messages, including both the new and old messages.
        """
        print(f"\n{'=' * 70}")
        print("🤖  assistant")
        print(f"{'-' * 70}")
        new_messages: list[dict[str, str]] = []

        # Get response
        tools = self.TOOLS if self.TOOLS else None
        response = self.model.generate(
            messages, tools=tools, max_new_tokens=self.max_new_tokens
        )
        new_messages.append({"role": "assistant", "content": response})

        # Parse tool calls
        tool_calls = self._parse_tool_calls(response)
        if not tool_calls:
            return [*messages, *new_messages]

        # Execute tool calls
        for tc in tool_calls:
            # Get function name and arguments
            fn_name = tc["name"]
            fn_args = tc.get("arguments", {})
            if isinstance(fn_args, str):
                fn_args = json.loads(fn_args)

            print(f"\n{'=' * 70}")
            print(f"🔨  tool {fn_name} {fn_args}")
            print(f"{'-' * 70}")

            # Get tool result
            result = (
                self.TOOL_REGISTRY[fn_name](**fn_args)
                if fn_name in self.TOOL_REGISTRY
                else f"Unknown tool: '{fn_name}'"
            )
            new_messages.append(
                {"role": "tool", "name": fn_name, "args": fn_args, "content": result}
            )
            print(f"{result}")

        return [*messages, *new_messages]

    @staticmethod
    def _parse_tool_calls(response: str) -> list[dict]:
        """Parse tool calls from model responses.

        Different models format their tool calls. This method is
        capable to parse both JSON-formatted and pseudo-XML-formatted
        tool calls.

        Args:
            response: The model response to be scanned for tool calls.

        Returns:
            A list of tool calls. Each tool call is a dictionary with
            a `name` and an `arguments` entry.
        """
        TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
        FUNCTION_NAME_PATTERN = re.compile(r"<function=(.*?)>")
        FUNCTION_BODY_PATTERN = re.compile(r"<function=.*?>.</function>", re.DOTALL)
        FUNCTION_ARGS_PATTERN = re.compile(
            r"<parameter=(.*?)>(.*?)</parameter>", re.DOTALL
        )

        tool_call_matches = re.findall(TOOL_CALL_PATTERN, response)
        tool_calls = []
        for raw_tc in tool_call_matches:
            # Parse JSON
            try:
                tool_calls.append(json.loads(raw_tc))
                continue
            except json.JSONDecodeError:
                pass

            # Parse fake XML
            try:
                name = re.findall(FUNCTION_NAME_PATTERN, raw_tc)[0]
                body = re.findall(FUNCTION_BODY_PATTERN, raw_tc)[0]
                args = re.findall(FUNCTION_ARGS_PATTERN, body)
                tool_call = {
                    "name": name.strip(),
                    "arguments": {
                        arg_name.strip(): arg.strip() for arg_name, arg in args
                    },
                }
                tool_calls.append(tool_call)
                continue
            except IndexError:
                pass

            # If the above attempts failed, the tool call is unparsable
            print(f"⚠️ UNPARSABLE TOOL CALL: {raw_tc}")
        return tool_calls

    @staticmethod
    def _parse_dep_tree(response: str) -> list[dict]:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        try:
            tree = re.findall(r"(\[{.+?}\])", response)[-1]
            return json.loads(tree)
        except json.JSONDecodeError:
            return []
        except IndexError:
            return []

    @staticmethod
    def _print_msg(msg: dict):
        """Pretty-print a message.

        Args:
            msg: A message in form of a dictionary.
        """
        role_emojis = {"system": "💻", "user": "👤", "assistant": "🤖", "tool": "🔨"}
        role = msg["role"]
        content = msg["content"]
        print(f"\n{'=' * 70}")
        print(f"{role_emojis[role]}  {role}")
        print(f"{'-' * 70}\n{content}")

    @staticmethod
    def _log_run(messages: list[dict]):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = Path(__file__).resolve().parent / "agent_logs"
        log_path = log_dir / f"run_{timestamp}.json"
        with open(log_path, "w") as f:
            json.dump(messages, f, indent=4)


# Just for testing purposes
def tree_parser(sentence: str):
    return "Unable to parse sentence completely, but the C3PO is definitly the root of the sentence and should be connected to `0`."


if __name__ == "__main__":
    # from agentic_nlp_pipeline import LocalModel

    # MODEL_ID = "Qwen/Qwen3.5-2B"
    # MODEL_ID = "Qwen/Qwen3-0.6B"
    # MODEL_ID = "Qwen/Qwen3-2B"
    # MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # JSON-formatted tool calls
    # model = LocalModel(MODEL_ID, device="xpu", enable_thinking=False)

    from agentic_nlp_pipeline.prompting.templates import DIRECT_PARSING_SYSTEM_PROMPT
    from agentic_nlp_pipeline import LlamaCppModel

    model = LlamaCppModel()

    agent = DepParseAgent(
        model,
        DIRECT_PARSING_SYSTEM_PROMPT
        + "\n\nUse should make at least one tool call!",  # (Wrap your tool calls into <tool_call></tool_call> tags!)",
        max_new_tokens=10000,
        max_iters_per_call=5,
    )

    tool = {
        "type": "function",
        "function": {
            "name": "tree_parser",
            "description": (
                "Evaluates a sentence and returns its dependency tree. "
                "Returns the tree as a new-line-separated list of edges. "
                "Example: `I am Luca.` -> (1, 3)\n(2, 3)\n(3, 0)\n(4, 3)"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sentence": {
                        "type": "string",
                        "description": "Any sentence, any format.",
                    }
                },
                "required": ["sentence"],
            },
        },
    }
    agent.register_tool(tool, tree_parser)

    sent1 = Sentence(
        [
            {"id": 1, "text": "My"},
            {"id": 2, "text": "name"},
            {"id": 3, "text": "is"},
            {"id": 4, "text": "C3PO"},
            {"id": 5, "text": "."},
        ]
    )
    sent1.text = " ".join([w.text for w in sent1.words])
    sent1.sent_id = "test-sentence-1"

    agent.run(sent=sent1)
    print(sent1)
