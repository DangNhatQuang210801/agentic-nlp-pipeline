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
        sent_id = sent.sent_id
        sent_text = sent.text
        sent_words = [{"ID": w.id, "FORM": w.text} for w in sent.words]

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

        # TODO: parse tree
        tree = self._parse_dep_tree(messages[-1]["content"])
        print(tree)
        # TODO: Create new sentence with HEAD attributes set to model output
        return sent

    def _run_iteration(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Run one iteration of obtaining a model response and making
        executing tool calls.

        Args:
            messages: The list of messages as context.

        Returns:
            A new list of messages, including both the new and old messages.
        """
        new_messages: list[dict[str, str]] = []

        # Get response
        tools = self.TOOLS if self.TOOLS else None
        response = self.model.generate(
            messages, tools=tools, max_new_tokens=self.max_new_tokens
        )
        new_messages.append({"role": "assistant", "content": response})
        self._print_msg(new_messages[-1])

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

            # Get tool result
            result = (
                self.TOOL_REGISTRY[fn_name](**fn_args)
                if fn_name in self.TOOL_REGISTRY
                else f"Unknown tool: '{fn_name}'"
            )
            new_messages.append(
                {"role": "tool", "name": fn_name, "args": fn_args, "content": result}
            )
            self._print_msg(new_messages[-1])

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
    def _parse_dep_tree(response: str) -> dict[int, int]:
        EDGE_PATTERN = re.compile(r"(\d+), ?(\d+)")
        edge_matches = re.findall(EDGE_PATTERN, response)
        return {s_node: e_node for s_node, e_node in edge_matches}

    @staticmethod
    def _print_msg(msg: dict):
        """Pretty-print a message.

        Args:
            msg: A message in form of a dictionary.
        """
        role_emojis = {"system": "💻", "user": "👤", "assistant": "🤖", "tool": "🔨"}
        role = msg["role"]
        name = msg.get("name", "")
        args = msg.get("args", "")
        content = msg["content"]
        print(
            f"\n{'=' * 60}"
            f"\n{role_emojis[role]}  {role} {name} {args}"
            f"\n{'—' * 60}"
            f"\n{content}"
        )

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
    from agentic_nlp_pipeline import LocalModel
    from agentic_nlp_pipeline.prompting.templates import DIRECT_PARSING_SYSTEM_PROMPT

    MODEL_ID = "Qwen/Qwen3.5-2B"
    # MODEL_ID = "Qwen/Qwen3-0.6B"
    # MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # JSON-formatted tool calls
    model = LocalModel(MODEL_ID, device="xpu")

    agent = DepParseAgent(
        model,
        DIRECT_PARSING_SYSTEM_PROMPT
        + "\nYou should always start with a tool call!",  # (Wrap your tool calls into <tool_call></tool_call> tags!)",
        max_new_tokens=500,
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
    print(sent1.text)

    parsed_sent = agent.run(sent=sent1)
    """output:
    System: You are a precise annotator assistent. Your purpose is to assist a corpus linguist in the creation of dependency trees.

    A dependency tree is a directed graph over a set of nodes (words) subject to the following conditions:
    1. It has a single node with no incoming edges. This is the root node. In the output, the root's head entry is written as 0, since it has no head of its own.
    2. Every other node has exactly one edge coming from its head.
    3. It is acyclic, so when following along the edges you never end up at the same node again.
    4. Each edge points from a dependency to its head, encoding the asymetric relation between the two.
    In linguistic terms, a head is the word that determines the grammatical and semantic core of a phrase or construction — it dictates the category of the whole unit, governs agreement,
    and is the element the rest of the phrase serves to modify, complete, or specify. For example, in 'the old book,' book is the head: it's what the phrase is fundamentally about, and 
    old and the merely add information about it. In 'she sleeps,' the verb sleeps is the head of the clause, since it determines the clause's core predication and licenses the subject.

    A dependency is the grammatical relation holding between a head and the word that depends on it — the dependent. The dependent is licensed, selected, or modified by its head: an adje
    ctive depends on the noun it describes, a subject depends on the verb it complements, a determiner depends on the noun it specifies. The dependency relation captures how the dependen
    t relates to its head — whether it's a modifier, an argument, a complement, and so on.

    For a given tokenized sentence, your task is to parse a dependency tree as described above. Sentence will be given to you in the form of a dictonary {<node_id>: '<word>'}. Your answe
    r MUST be given in the following format:
    - One line for every word in the input
    - One tuple in every line
    - The first entry must be the node id of the dependecy
    - The second entry must be the node id of the head
    A few practical hints:
    - No node is its own head.
    - Only one word (the root) can have 0 as its head.
    - By convention, sentence final punctuation attaches to the root.

    You should always start with a tool call!
    User: {1: 'My', 2: 'name', 3: 'is', 4: 'C3PO', 5: '.'}
    ============================================================

    🤖  Response: <tool_call>
    {"name": "tree_parser", "arguments": {"sentence": "My name is C3PO."}}
    </tool_call>

    🔨  Tool call  : tree_parser({'sentence': 'My name is C3PO.'})
    Tool result: Unable to parse sentence completely, but the C3PO is definitly the root of the sentence and should be connected to `0`.

    🤖  Response: The sentence "My name is C3PO." cannot be parsed completely due to missing words. However, based on the structure you've provided, C3PO is definitively the root of the 
    sentence and should be connected to the root node (0).
    """
