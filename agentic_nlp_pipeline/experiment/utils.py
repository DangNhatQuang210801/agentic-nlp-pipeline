from pathlib import Path

from stanza.models.common.doc import Sentence


def get_unparsed_sentences(
    root_dir: Path, sup_suffix: str, sub_suffix: str, limit: int | None = None,
) -> list[Path]:
    """Given a parent directory, return all those files with suffix
    `sup_suffix` such that there is no file with the same name but
    `sub_suffix` in place of `sup_suffix`.

    Example: Get all CoNLL-U files that have not yet been dependency
    parsed as indicated by the file name suffixes.

    Args:
        root_dir: The directory in which to iterate over directories.
        sup_suffix: The file name suffix of the files of interest.
        sub_suffix: The file name suffix that signals 'already done'.

    Yields:
        Paths of unprocessed files.
    """

    # The following is a special case for a single language instead of all langauges:
    if any(root_dir.rglob(f"*{sup_suffix}")):
        return _get_diffset_of_paths(root_dir, sup_suffix, sub_suffix)

    all_unparsed_sents = []
    for dir in root_dir.iterdir():
        if not dir.is_dir() or len(dir.name) != 3:
            continue
        all_unparsed_sents.extend(_get_diffset_of_paths(dir, sup_suffix, sub_suffix))
    # return all_unparsed_sents

    # determines the number of sentences to return:
    if limit is not None:
        return all_unparsed_sents[:limit]
    return all_unparsed_sents


def _get_diffset_of_paths(dir: Path, sup_suffix: str, sub_suffix: str) -> list[Path]:
    """Get file paths to unprocessed files.

    Args:
        dir: The directory in which to look for matching files.
        sup_suffix: The file name suffix of the files of interest.
        sub_suffix: The file name suffix that signals 'already done'.

    Yields:
        Paths of unprocessed files.
    """
    sup_paths = get_paths_by_suffix(dir, sup_suffix)
    sub_paths = get_paths_by_suffix(dir, sub_suffix)
    sup_sent_ids = {str(path).removesuffix(sup_suffix) for path in sup_paths}
    sub_sent_ids = {str(path).removesuffix(sub_suffix) for path in sub_paths}
    return [Path(sent_id + sup_suffix) for sent_id in sup_sent_ids - sub_sent_ids]


def get_paths_by_suffix(dir: Path, suffix: str) -> list[Path]:
    """Get file paths with a certain suffix.

    Args:
        dir: The directory in which to look for matching files.
        suffix: The file name suffix of the files of interest.

    Yields:
        Paths of matchinf files.
    """
    return list(dir.rglob(f"*{suffix}"))


def clear_heads(sent: Sentence):
    """Remove the HEAD attribute of all words of a sentence.

    Args:
        sent: A Stanza Sentence object.
    """
    for word in sent.words:
        word.head = None
