import ast
from datetime import datetime
import json
from pathlib import Path
from typing import Callable, Any
from urllib import response

import regex as re
from stanza.models.common.doc import Sentence
from ..models import LanguageModel


# Some of this code is loosely based on the exercise to Lecture 07.
class DepParseAgent:
    TOOLS: list[dict] = []
    TOOL_REGISTRY: dict[str, Callable] = {}
    stats: dict[str, dict[str, Any]] = {}

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
            method: The function associated with the tool.
        """
        self.TOOLS.append(tool_schema)
        self.TOOL_REGISTRY[tool_schema["function"]["name"]] = method

    def dep_parse(
        self,
        sent: Sentence,
        log_dir: Path | None = None,
        log_file_name: str | None = None,
    ):
        """Parse the dependency tree of a sentence.

        Args:
            sent: A Stanza Sentence object.
            log_dir: Optional Path to a logging directory.
            log_file_name: Optional name for the log file.
        """
        sent_text = sent.text
        sent_words = [{"id": w.id, "form": w.text} for w in sent.words]
        sent_words_str = re.sub(r"'", '"', str(sent_words))

        content = f"Sentence: {sent_text}\nStructure:\n{sent_words_str}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]

        for msg in messages:
            self._print_msg(msg)

        for _ in range(self.max_iters_per_call):
            messages = self._run_iteration(messages)
            self._update_stats(messages)
            if messages[-1]["role"] != "tool":
                break

        self._log_run(messages, log_dir, log_file_name)
        self._update_stats(messages)

        tree = self._parse_dep_tree(messages[-1]["content"])
        for word, edge in zip(sent.words, tree):
            word.head = edge["head"]

    def _run_iteration(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """One iteration of obtaining a model response and
        executing tool calls.

        Args:
            messages: A list of message dicts.

        Returns:
            A new list of message dicts, including both the new and old messages.
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
        # added:

        print("\n===== RAW MODEL OUTPUT =====")
        print(response)
        print("============================")


        if not tool_calls:
            return [*messages, *new_messages]

        # Execute tool calls
        for tc in tool_calls:
            # Get function name and arguments
            fn_name = tc["name"]
            fn_args = tc.get("arguments", {})

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

        _decoder = json.JSONDecoder()

        def _parse_arg(value: Any, _depth: int = 0):
            # If value is not a string, return
            if not isinstance(value, str) or _depth > 3:
                return value
            
            stripped = value.strip()
            parsed = None
            
            # Try strict parse first
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                # Fall back to raw_decode: parses the leading valid JSON value
                # and ignores trailing junk (e.g. a stray extra "]" or "}")    
                try:
                    parsed, end_idx = _decoder.raw_decode(stripped)
                    leftover = stripped[end_idx:].strip()
                    if leftover:
                        print(f"⚠️ Ignored trailing garbage after JSON arg: {leftover!r}")
                except json.JSONDecodeError:
                    try:
                        parsed = ast.literal_eval(stripped)
                    except (ValueError, SyntaxError):
                        return value 
            
            
            # If we only unwrapped one layer of quoting and still have a string
            # that looks like JSON/Python literal, try again.
            if isinstance(parsed, str) and parsed != value:
                return _parse_arg(parsed, _depth + 1)
            return parsed        
        

        TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
        FUNCTION_NAME_PATTERN = re.compile(r"<function=(.*?)>")
        FUNCTION_BODY_PATTERN = re.compile(r"<function=.*?>(.*?)</function>", re.DOTALL)
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
                        arg_name.strip(): _parse_arg(arg.strip())
                        for arg_name, arg in args
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
        """Parse the dependency tree from the model output.

        Args:
            response: The model response as str to parse the
                dependency tree from.

        Returns:
            The dependency tree parsed as a list of dicts.
        """
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        try:
            tree = re.findall(r"(\[{.+?}\])", response)[-1]
            return json.loads(tree)
        except (json.JSONDecodeError, IndexError):
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
    def _log_run(
        messages: list[dict], log_dir: Path | None = None, file_name: str | None = None
    ):
        """Dump the list of message dicts into a json file.

        Args:
            messages: A list of message dicts.
        """
        # If no arguments are provided, set defaults
        if log_dir is None:
            log_dir = Path(__file__).resolve().parent / "agent_logs"
        if file_name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = f"run_{timestamp}"

        log_path = log_dir / (file_name + ".json")
        with open(log_path, "w") as f:
            json.dump(messages, f, indent=4)

    def _update_stats(self, messages: list[dict[str, Any]]):
        """Compute per-role message statistics.

        Args:
            messages: A list of message dicts.
        """
        self.stats = {}
        for msg in messages:
            role = msg["role"]
            role_dict = self.stats.setdefault(role, {})
            role_dict["n"] = role_dict.get("n", 0) + 1
            if role == "tool":
                name = msg["name"]
                tools_dict = role_dict.setdefault("tools", {})
                tools_dict[name] = tools_dict.get("n", 0) + 1
