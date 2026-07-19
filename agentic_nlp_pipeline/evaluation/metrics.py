import json
from pathlib import Path
from typing import overload

import pandas as pd
import regex as re
from stanza.models.common.doc import Document, Sentence
from stanza.utils.conll import CoNLL
from transformers import AutoTokenizer

from ..experiment.utils import get_paths_by_suffix
from ..validation import validate_sentence
from .projectivity import is_projective


def compile_df(
    repo_root: Path,
    gold_suffix: str,
    pred_suffix: str,
    log_suffix: str,
    with_tools: bool,
    tokenizer: AutoTokenizer,
) -> pd.DataFrame:
    data = {
        "sent_id": [],
        "language": [],
        "n_words": [],
        "n_correct": [],
        "uas": [],
        "original_projective": [],
        "parsed_successfully": [],
        "has_valid_tree_structure": [],
        "with_tools": [],
        "n_tool_calls": [],
        "lookup_morphology": [],
        "retrieve_similar_sentences": [],
        "retrieve_similar_sentences_bow": [],
        "validate_dependency_tree": [],
        "n_generated_tokens": [],
        "n_tool_result_tokens": [],
    }

    for language in ("eng", "nds", "nan", "mar", "vie"):
        data_root = repo_root / "data" / "processed" / language

        # Fetch file paths of parsed and unparsed sentences
        unparsed = get_paths_by_suffix(data_root, gold_suffix)
        parsed = get_paths_by_suffix(data_root, pred_suffix)
        logs = get_paths_by_suffix(data_root, log_suffix)

        # Turn lists of paths into dicts
        unparsed_dict = _create_path_dict(unparsed, gold_suffix)
        parsed_dict = _create_path_dict(parsed, pred_suffix)
        logs_dict = _create_path_dict(logs, log_suffix)

        for name, parsed_path in parsed_dict.items():
            matching_unparsed_path = unparsed_dict[name]
            matching_log_path = logs_dict[name]

            gold_sent = CoNLL.conll2doc(matching_unparsed_path).sentences[0]
            pred_sent = CoNLL.conll2doc(parsed_path).sentences[0]
            message_log = json.load(open(matching_log_path, "r"))

            n_words = len(pred_sent.words)
            n_correct = _get_n_correct(gold_sent, pred_sent)
            original_projective = is_projective(gold_sent)
            parsed_successfully = _was_parsed_successfully(pred_sent)
            n_tool_calls = _get_n_tool_calls(message_log)

            data["sent_id"].append(pred_sent.sent_id)
            data["language"].append(language)
            data["n_words"].append(n_words)
            data["n_correct"].append(n_correct)
            data["uas"].append(n_correct / n_words)
            data["original_projective"].append(original_projective)
            data["parsed_successfully"].append(parsed_successfully)
            data["has_valid_tree_structure"].append(validate_sentence(pred_sent)[0])
            data["with_tools"].append(with_tools)
            data["n_tool_calls"].append(n_tool_calls)
            data["lookup_morphology"].append(
                len(re.findall(r"'lookup_morphology'", str(message_log)))
            )
            data["retrieve_similar_sentences"].append(
                len(re.findall("'retrieve_similar_sentences'", str(message_log)))
            )
            data["retrieve_similar_sentences_bow"].append(
                len(re.findall("'retrieve_similar_sentences_bow'", str(message_log)))
            )
            data["validate_dependency_tree"].append(
                len(re.findall("'validate_dependency_tree'", str(message_log)))
            )
            data["n_generated_tokens"].append(
                _get_content_length(message_log, "assistant", tokenizer)
            )
            data["n_tool_result_tokens"].append(
                _get_content_length(message_log, "tool", tokenizer)
            )

    return pd.DataFrame(data)


def _create_path_dict(paths: list[Path], suffix: str) -> dict[str, Path]:
    return {str(path).removesuffix(suffix): path for path in paths}


def _get_content_length(messages: list[dict], role: str, tokenizer) -> int:
    acc_content = ""
    for msg in messages:
        if msg["role"] == role:
            acc_content += msg["content"]
    length = len(tokenizer.encode(acc_content, add_special_tokens=False))
    return length


def _get_n_tool_calls(messages: list[dict]) -> int:
    return sum(1 for msg in messages if msg["role"] == "tool")


def _was_parsed_successfully(sent: Sentence) -> bool:
    for i, word in enumerate(sent.words):
        if word.head != i:
            return True
    return False


@overload
def _get_n_correct(gold: Sentence, pred: Sentence) -> int: ...
@overload
def _get_n_correct(gold: Document, pred: Document) -> int: ...
def _get_n_correct(gold: Sentence | Document, pred: Sentence | Document) -> int:
    """Compute the unlabeled attachment score (UAS) for pairs of
    sentences or documents.

    Args:
        gold: The sentence or document to be taken as the standard
        to compare agains.
        pred: The sentence or document that should be tested.

    Returns:
        The unlabeled attachment score (UAS).
    """
    if isinstance(gold, Sentence):
        gold = Document([gold.to_dict()])
    if isinstance(pred, Sentence):
        pred = Document([pred.to_dict()])

    assert isinstance(gold, Document)
    assert isinstance(pred, Document)
    assert len(gold.sentences) == len(pred.sentences)

    n_correct = 0
    for gold_sent, pred_sent in zip(gold.sentences, pred.sentences):
        assert len(gold_sent.words) == len(pred_sent.words)
        for gold_word, pred_word in zip(gold_sent.words, pred_sent.words):
            if gold_word.head == pred_word.head:
                n_correct += 1

    return n_correct
