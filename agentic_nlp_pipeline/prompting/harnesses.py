from stanza.models.common.doc import Sentence

from agentic_nlp_pipeline import LocalModel
from templates import DIRECT_PARSING_SYSTEM_PROMPT


def _sent_to_dict(sent: Sentence) -> dict[int, str]:
    return {word.id: word.text for word in sent.words}


def direct_parsing(model, sent: Sentence, max_new_tokens: int):
    content = str(_sent_to_dict(sent))

    messages = [
        {"role": "system", "content": DIRECT_PARSING_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    return model.generate(messages, max_new_tokens)


if __name__ == "__main__":
    # MODEL_ID = "Qwen/Qwen3-0.6B"
    MODEL_ID = "Qwen/Qwen3-2B"
    model = LocalModel(MODEL_ID)

    sent1 = Sentence(
        [
            {"id": 1, "text": "My"},
            {"id": 2, "text": "name"},
            {"id": 3, "text": "is"},
            {"id": 4, "text": "C3PO"},
            {"id": 5, "text": "."},
        ]
    )

    reply = direct_parsing(model, sent1, 500)

    print(f"\n🖥️ System: {DIRECT_PARSING_SYSTEM_PROMPT}")
    print(f"👨‍🦲 User: {_sent_to_dict(sent1)}")
    print(f"🤖 Bot : {reply}")
