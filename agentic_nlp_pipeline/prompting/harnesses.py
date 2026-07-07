from stanza.models.common.doc import Sentence

from agentic_nlp_pipeline import LocalModel, LanguageModel
from templates import DIRECT_PARSING_SYSTEM_PROMPT


def _sent_to_dict(sent: Sentence) -> dict[int, str]:
    """Convert a CoNLL-U Sentence into a {id: text} dict.

    Args:
        sent: Some Stanza Sentence.

    Returns:
        A dictionary of the form {id: text}, where `id` is
        the word ID and `text` is the words FORM.
    """
    return {word.id: word.text for word in sent.words}


def direct_parsing(model: LanguageModel, sent: Sentence, max_new_tokens: int):
    """One-shot dependency parsing.

    This harness is meant for direct dependency parsing: Give the
    model a sentence and ask it to produce a dependency tree in one
    shot. The model may use thinking, but the final answer must be
    a complete tree. This tree is expected to be encoded as a new-
    line-separated list of tuples of the form (ID, HEAD).

    TODO: Right now it is designed to only predict the edges (HEAD)
          but perhaps we should also include the relations (DEPREL).

    Args:
        model: Some LanguageModel.
        sent: Some Stanza Sentence.
        max_new_tokens: The maximum number of new tokens to be generated.

    Returns:
        (right now it's a string, later it should be a Stanza Sentence.
    """
    content = str(_sent_to_dict(sent))

    messages = [
        {"role": "system", "content": DIRECT_PARSING_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    # TODO: Parse output

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
