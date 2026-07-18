from stanza.models.common.doc import Document, Sentence


def is_projective(sent: Sentence) -> bool:
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
        if is_projective(sent):
            n_projective += 1
    return n_projective / len(doc.sentences)
