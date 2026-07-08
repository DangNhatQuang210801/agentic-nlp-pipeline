"""Tests for smaller utility functions."""

from stanza.models.common.doc import Document, Sentence

from agentic_nlp_pipeline.evaluation.projectivity import isprojective, projectivity_rate
from agentic_nlp_pipeline.validation.tree_validation import (
    _has_single_root,
    _is_acyclic,
    _has_valid_heads,
)


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
