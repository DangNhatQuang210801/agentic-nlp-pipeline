import json
from typing import overload

import regex as re
from stanza.models.common.doc import Sentence

from agentic_nlp_pipeline import LanguageModel


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

    def _sent_to_dict(self, sent: Sentence) -> dict[int, str]:
        """Convert a CoNLL-U Sentence into a {id: text} dict.

        Args:
            sent: Some Stanza Sentence.

        Returns:
            A dictionary of the form {id: text}, where `id` is
            the word ID and `text` is the words FORM.
        """
        return {word.id: word.text for word in sent.words}

    def _parse_tool_calls(self, response: str):
        pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"  # parse the tool call by <tool_call> and </tool_call>
        matches = re.findall(pattern, response, re.DOTALL)
        tool_calls = []
        for raw in matches:
            try:
                tool_calls.append(json.loads(raw))
            except json.JSONDecodeError:
                pass
        return tool_calls

    def register_tool(self, tool, method):
        self.TOOLS.append(tool)
        self.TOOL_REGISTRY[tool["function"]["name"]] = method

    # @overload
    # def register_tools(self, tools: dict): ...
    # @overload
    # def register_tools(self, tools: list[dict]): ...
    # def register_tools(self, tools: dict | list[dict]):
    #     if isinstance(tools, dict):
    #         self.TOOL_REGISTRY.append(tools)
    #     if isinstance(tools, list):
    #         self.TOOL_REGISTRY.extend(tools)

    # Code inspired by the exercise from Lecture 07
    def run(self, sent: Sentence):
        content = str(self._sent_to_dict(sent))

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]

        print(f"{'=' * 60}")
        print(f"  System: {self.system_prompt}")
        print(f"  User: {content}")
        print(f"{'=' * 60}")

        for _ in range(self.max_iters_per_call):
            tools = self.TOOLS if len(self.TOOLS) > 0 else None
            response = self.model.generate(
                messages, tools=tools, max_new_tokens=self.max_new_tokens
            )
            print(f"\n🤖  Response: {response}")

            tool_calls = self._parse_tool_calls(response)
            if not tool_calls:
                answer = re.sub(r"<[^>]+>", "", response).strip()
                print(f"\n🤖  Answer: {answer}")
                return answer

            messages.append({"role": "assistant", "content": response})

            for tc in tool_calls:
                fn_name = tc.get("name")
                fn_args = tc.get("arguments", {})
                if isinstance(fn_args, str):
                    fn_args = json.loads(fn_args)

                print(f"\n🔨  Tool call  : {fn_name}({fn_args})")

                result = (
                    self.TOOL_REGISTRY[fn_name](**fn_args)
                    if fn_name in self.TOOL_REGISTRY
                    else f"Unknown tool: '{fn_name}'"
                )
                print(f"  Tool result: {result}")

                messages.append({"role": "tool", "name": fn_name, "content": result})


# Just for testing purposes
def tree_parser(sentence: str):
    return "Unable to parse sentence completely, but the C3PO is definitly the root of the sentence and should be connected to `0`."


if __name__ == "__main__":
    from agentic_nlp_pipeline import LocalModel
    from agentic_nlp_pipeline.prompting.templates import DIRECT_PARSING_SYSTEM_PROMPT

    # The different models seem to format tool calls differently (json vs. xml style).
    # Currently, only Qwen2.5 is handled properly.
    # TODO: make _parse_tool_calls more robust.

    # MODEL_ID = "Qwen/Qwen3.5-2B"
    # MODEL_ID = "Qwen/Qwen3-0.6B"
    MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
    model = LocalModel(MODEL_ID)

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
