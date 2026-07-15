from pathlib import Path
from typing import Generator

from stanza.models.common.doc import Sentence
from stanza.utils.conll import CoNLL

from agentic_nlp_pipeline import DepParseAgent, LlamaCppModel
from agentic_nlp_pipeline.prompting import templates

UNPARSED = "--original.conllu"
PARSED_DIRECTLY = "--parsed-d.conllu"
PARSED_AGENTICALLY = "--parsed-a.conllu"


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data" / "processed"

    model = LlamaCppModel()
    agent = DepParseAgent(model, templates.DIRECT_PARSING_SYSTEM_PROMPT, 10000, 10)

    for sent_path in _get_unparsed_sentences(data_root, UNPARSED, PARSED_DIRECTLY):
        doc = CoNLL.conll2doc(sent_path)
        sent = doc.sentences[0]
        _clear_heads(sent)
        agent.parse(sent)


def _get_unparsed_sentences(
    root_dir: Path, sup_suffix: str, sub_suffix: str
) -> Generator[Path]:
    for dir in root_dir.iterdir():
        if not dir.is_dir() or len(dir.name) != 3:
            continue
        for sent_path in _get_diffset_of_sentences(dir, sup_suffix, sub_suffix):
            yield sent_path


def _get_diffset_of_sentences(
    dir: Path, sup_suffix: str, sub_suffix: str
) -> Generator[Path]:
    sup_paths = list(dir.rglob(sup_suffix))
    sub_paths = list(dir.rglob(sub_suffix))
    sup_sent_ids = {str(path).removesuffix(sup_suffix) for path in sup_paths}
    sub_sent_ids = {str(path).removesuffix(sub_suffix) for path in sub_paths}
    for sent_id in sup_sent_ids - sub_sent_ids:
        yield Path(sent_id + sup_suffix)


def _clear_heads(sent: Sentence):
    for word in sent.words:
        word.head = None


if __name__ == "__main__":
    main()
