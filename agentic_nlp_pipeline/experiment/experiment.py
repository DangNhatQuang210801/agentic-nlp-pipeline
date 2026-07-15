from pathlib import Path

from stanza.utils.conll import CoNLL

from . import utils
from ..prompting import DepParseAgent

UNPARSED = "--original.conllu"
PARSED_DIRECTLY = "--parsed-directly.conllu"
PARSED_AGENTICALLY = "--parsed-agentically.conllu"


class Experiment:
    def __init__(self, agent: DepParseAgent, data_root: Path, new_suffix: str):
        self.agent = agent
        self.data_root = data_root
        self.new_suffix = new_suffix

    def run(self):
        for sent_path in utils.get_unparsed_sentences(
            self.data_root, UNPARSED, self.new_suffix
        ):
            doc = CoNLL.conll2doc(sent_path)
            sent = doc.sentences[0]
            utils.clear_heads(sent)
            self.agent.dep_parse(
                sent=sent,
                log_dir=sent_path.parent,
                log_file_name=str(sent_path).removesuffix(UNPARSED) + "--log",
            )

            # Write parsed sentence to disk
            new_file_path = Path(
                str(sent_path).removesuffix(UNPARSED) + self.new_suffix
            )
            CoNLL.write_doc2conll(doc, new_file_path)
