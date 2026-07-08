"""Tests for smaller utility functions."""

from stanza.models.common.doc import Document, Sentence

from agentic_nlp_pipeline.evaluation.projectivity import isprojective, projectivity_rate


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
