from copy import deepcopy
import regex as re
from stanza.models.common.doc import Sentence

EDGE = r"\(\d+, ?\d+\)"
TREE = rf"{EDGE}(?:\n{EDGE})*"
S_E_NODES = r"\((\d+), ?(\d+)\)"


def parse_tree_from_text(text: str) -> list[tuple[int, int]]:
    try:
        tree = re.findall(TREE, text)[-1]
    except IndexError:
        return []

    edges = []
    for s_node, e_node in re.findall(S_E_NODES, tree):
        edges.append((int(s_node), int(e_node)))
    return edges


def insert_tree_into_sentence(sent: Sentence, tree: list[tuple[int, int]]) -> Sentence:
    sent = deepcopy(sent)
    for s_node, e_node in tree:
        sent.words[s_node - 1].head = e_node - 1
    return sent


if __name__ == "__main__":
    text = "(1, 2)\n(3,4)\n(5, s)"
    parsed = parse_tree_from_text(text)
    print(parsed)
