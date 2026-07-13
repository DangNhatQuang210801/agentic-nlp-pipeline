"""Script for testing the agent."""

import json

from stanza.models.common.doc import Sentence

from agentic_nlp_pipeline import DepParseAgent
from agentic_nlp_pipeline import KNNRetrievalTool, TreeValidationTool
from agentic_nlp_pipeline.prompting.templates import DIRECT_PARSING_SYSTEM_PROMPT
from agentic_nlp_pipeline import LlamaCppModel


# Mock tool
def tree_parser(sentence: str):
    sentence = sentence
    return "Unable to parse sentence completely, but the C3PO is definitly the root of the sentence and should be connected to `0`."


if __name__ == "_main__":
    model = LlamaCppModel()

    agent = DepParseAgent(
        model,
        DIRECT_PARSING_SYSTEM_PROMPT
        + "\n\nUse should make at least one tool call!",  # (Wrap your tool calls into <tool_call></tool_call> tags!)",
        max_new_tokens=10000,
        max_iters_per_call=5,
    )

    with open(
        "agentic_nlp_pipeline/prompting/agent_logs/run_2026-07-12_22-52-55.json", "r"
    ) as f:
        raw = f.read()
    messages = json.loads(raw)
    agent._update_stats(messages)
    print(agent.stats)

if __name__ == "__main__":
    # from agentic_nlp_pipeline import LocalModel

    # MODEL_ID = "Qwen/Qwen3.5-2B"
    # MODEL_ID = "Qwen/Qwen3-0.6B"
    # MODEL_ID = "Qwen/Qwen3-2B"
    # MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # JSON-formatted tool calls
    # model = LocalModel(MODEL_ID, device="xpu", enable_thinking=False)

    model = LlamaCppModel()

    agent = DepParseAgent(
        model,
        DIRECT_PARSING_SYSTEM_PROMPT,  # + "\n\nUse should make at least one tool call!",  # (Wrap your tool calls into <tool_call></tool_call> tags!)",
        max_new_tokens=10000,
        max_iters_per_call=5,
    )

    mock_tool = {
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
    # agent.register_tool(mock_tool, tree_parser)
    agent.register_tool(*TreeValidationTool.as_agent_tool())

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
    print(agent.stats)
