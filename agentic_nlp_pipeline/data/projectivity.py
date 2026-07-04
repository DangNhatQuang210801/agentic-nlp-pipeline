from typing import Optional
from stanza.models.common.doc import Sentence, Document


def isprojective(sent: Sentence) -> Optional[bool]:
    """Check if a sentence has a projective tree structure or not.

    A tree structure is non-projective if it cannot be drawn without
    some arcs crossing over one another like this:

      .–––––––––.
     /.–––. .–––|–––––.
    |/    |/    |     |
    A     B     C     D

    Args:
        sent: Some Stanza sentence.

    Returns:
        True for `projective` and False for `non-projective`.
    """
    for word in sent.words:
        lower = min(word.id, word.head)
        upper = max(word.id, word.head)
        for w in sent.words:
            if not lower < w.id < upper:
                continue
            if w.head < lower or w.head > upper:
                return False
    return True


def projectivity_rate(doc: Document) -> float:
    """Calculate the ratio of projective tree structures in `doc`.

    Args:
        doc: Some Stanza document.

    Returns:
        Ratio of projective trees.
    """
    n_projective = 0
    for sent in doc.sentences:
        if isprojective(sent):
            n_projective += 1
    return n_projective / len(doc.sentences)


# Simple test
if __name__ == "__main__":
    sent1 = Sentence(
        [
            {"id": 1, "text": "A", "head": 0},
            {"id": 2, "text": "B", "head": 1},
            {"id": 3, "text": "C", "head": 2},
            {"id": 4, "text": "D", "head": 3},
        ]
    )
    sent2 = Sentence(
        [
            {"id": 1, "text": "A", "head": 0},
            {"id": 2, "text": "B", "head": 1},
            {"id": 3, "text": "C", "head": 4},
            {"id": 4, "text": "D", "head": 2},
        ]
    )
    sent3 = Sentence(
        [
            {"id": 1, "text": "A", "head": 0},
            {"id": 2, "text": "B", "head": 1},
            {"id": 3, "text": "C", "head": 1},
            {"id": 4, "text": "D", "head": 2},
        ]
    )
    sent4 = Sentence(
        [
            {"id": 1, "text": "A", "head": 0},
            {"id": 2, "text": "B", "head": 4},
            {"id": 3, "text": "C", "head": 1},
            {"id": 4, "text": "D", "head": 1},
        ]
    )
    assert isprojective(sent1)
    assert isprojective(sent2)
    assert not isprojective(sent3)
    assert not isprojective(sent4)

    sentences = [sent1, sent2, sent3, sent4]
    doc = Document([sent.to_dict() for sent in sentences])
    assert projectivity_rate(doc) == 0.5

    print("All tests passed!")
