"""Tests for smaller utility functions."""

import json

from stanza.models.common.doc import Document, ID, Sentence, TEXT, UPOS

from agentic_nlp_pipeline.prompting.agent import DepParseAgent
from agentic_nlp_pipeline.evaluation.projectivity import isprojective, projectivity_rate
from agentic_nlp_pipeline.validation.tree_validation import (
    _has_single_root,
    _is_acyclic,
    _has_valid_heads,
)
from agentic_nlp_pipeline.tools.utils import sentence_to_token_dicts
from agentic_nlp_pipeline.tools import KNNRetrievalTool


# ====================================================================
#  projectivity.py
# ====================================================================


# projective
sent1 = Sentence(
    [
        {"id": 1, "text": "A", "head": 0},
        {"id": 2, "text": "B", "head": 1},
        {"id": 3, "text": "C", "head": 2},
        {"id": 4, "text": "D", "head": 3},
    ]
)
# projective
sent2 = Sentence(
    [
        {"id": 1, "text": "A", "head": 0},
        {"id": 2, "text": "B", "head": 1},
        {"id": 3, "text": "C", "head": 4},
        {"id": 4, "text": "D", "head": 2},
    ]
)
# not projective
sent3 = Sentence(
    [
        {"id": 1, "text": "A", "head": 0},
        {"id": 2, "text": "B", "head": 1},
        {"id": 3, "text": "C", "head": 1},
        {"id": 4, "text": "D", "head": 2},
    ]
)
# not projective
sent4 = Sentence(
    [
        {"id": 1, "text": "A", "head": 0},
        {"id": 2, "text": "B", "head": 4},
        {"id": 3, "text": "C", "head": 1},
        {"id": 4, "text": "D", "head": 1},
    ]
)


def test_isprojective():
    assert isprojective(sent1)
    assert isprojective(sent2)
    assert not isprojective(sent3)
    assert not isprojective(sent4)


def test_projectivity_rate():
    sentences = [sent1, sent2]
    doc = Document([sent.to_dict() for sent in sentences])
    assert projectivity_rate(doc) == 1.0

    sentences = [sent3, sent4]
    doc = Document([sent.to_dict() for sent in sentences])
    assert projectivity_rate(doc) == 0.0

    sentences = [sent1, sent2, sent3, sent4]
    doc = Document([sent.to_dict() for sent in sentences])
    assert projectivity_rate(doc) == 0.5


# ====================================================================
#  tree_validation.py
# ====================================================================


# One HEAD out of range
sent5 = Sentence(
    [
        {"id": 1, "text": "A", "head": 0},
        {"id": 2, "text": "B", "head": 1},
        {"id": 3, "text": "C", "head": 2},
        {"id": 4, "text": "D", "head": 6},
    ]
)
# Multiple HEADs out of range
sent6 = Sentence(
    [
        {"id": 1, "text": "A", "head": 0},
        {"id": 2, "text": "B", "head": 7},
        {"id": 3, "text": "C", "head": 1},
        {"id": 4, "text": "D", "head": 6},
    ]
)
# No head
sent7 = Sentence(
    [
        {"id": 1, "text": "A", "head": 3},
        {"id": 2, "text": "B", "head": 1},
        {"id": 3, "text": "C", "head": 2},
        {"id": 4, "text": "D", "head": 3},
    ]
)
# Too many heads
sent8 = Sentence(
    [
        {"id": 1, "text": "A", "head": 0},
        {"id": 2, "text": "B", "head": 1},
        {"id": 3, "text": "C", "head": 2},
        {"id": 4, "text": "D", "head": 0},
    ]
)
# Single isolated node
sent9 = Sentence(
    [
        {"id": 1, "text": "A", "head": 0},
        {"id": 2, "text": "B", "head": 1},
        {"id": 3, "text": "C", "head": 1},
        {"id": 4, "text": "D", "head": 4},
    ]
)
# Cyclic graph
sent10 = Sentence(
    [
        {"id": 1, "text": "A", "head": 0},
        {"id": 2, "text": "B", "head": 4},
        {"id": 3, "text": "C", "head": 2},
        {"id": 4, "text": "D", "head": 3},
    ]
)
# Good sentence
sent11 = Sentence(
    [
        {"id": 1, "text": "A", "head": 0},
        {"id": 2, "text": "B", "head": 1},
        {"id": 3, "text": "C", "head": 2},
        {"id": 4, "text": "D", "head": 3},
    ]
)


def test_has_valid_head():
    """Check if sentence has valid HEAD"""
    assert not _has_valid_heads(sent5)[0]
    assert not _has_valid_heads(sent6)[0]
    assert _has_valid_heads(sent11)[0]


def test_has_single_root():
    """Check if sentence has a single root"""
    assert not _has_single_root(sent7)[0]
    assert not _has_single_root(sent8)[0]
    assert _has_single_root(sent11)[0]


def test_is_acyclic():
    """Check for cycles"""
    assert not _is_acyclic(sent9)[0]
    assert not _is_acyclic(sent10)[0]
    assert _is_acyclic(sent11)[0], _is_acyclic(sent11)[1]


# ====================================================================
#  agent.py
# ====================================================================


def test_parse_tool_calls():
    json_tool_call = """<tool_call>
    {"name": "tree_parser", "arguments": {"sentence": "My name is C3PO."}}
    </tool_call>"""
    xml_tool_call = """<tool_call>
<function=tree_parser>
<parameter=sentence>
My name is C3PO.
</parameter>
</function>
</tool_call>"""
    expected = [{"name": "tree_parser", "arguments": {"sentence": "My name is C3PO."}}]
    assert DepParseAgent._parse_tool_calls(json_tool_call) == expected
    assert DepParseAgent._parse_tool_calls(xml_tool_call) == expected


def test_sentence_to_token_dicts():
    sent = Sentence([{ID: 1, TEXT: "Dogs", UPOS: "NOUN"}])

    assert sentence_to_token_dicts(sent, fields=("id", "form", "upos")) == [
        {"id": 1, "form": "Dogs", "upos": "NOUN"}
    ]


# ====================================================================
#  retrieval.py
# ====================================================================


def test_knn_retrieval_tool(tmp_path):
    train = tmp_path / "toy.conllu"
    train.write_text(
        """# sent_id = s1
# text = I like cats
1	I	I	PRON	_	_	2	nsubj	_	_
2	like	like	VERB	_	_	0	root	_	_
3	cats	cat	NOUN	_	_	2	obj	_	_

# sent_id = s2
# text = Birds fly
1	Birds	bird	NOUN	_	_	2	nsubj	_	_
2	fly	fly	VERB	_	_	0	root	_	_
""",
        encoding="utf-8",
    )
    tool = KNNRetrievalTool.from_conllu_files({"english": train})

    query = Sentence(
        [
            {ID: 1, TEXT: "I", UPOS: "PRON"},
            {ID: 2, TEXT: "like", UPOS: "VERB"},
            {ID: 3, TEXT: "dogs", UPOS: "NOUN"},
        ]
    )
    results = tool.retrieve("english", query, k=1)

    assert results[0][1].sent_id == "s1"
    assert results[0][1].words[1].deprel == "root"

    tool_result = json.loads(
        tool.search(
            "english",
            [
                {"id": 1, "form": "I", "upos": "PRON"},
                {"id": 2, "form": "like", "upos": "VERB"},
            ],
            k=1,
        )
    )
    assert tool_result[0]["sent_id"] == "s1"
