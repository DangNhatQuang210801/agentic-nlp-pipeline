from pathlib import Path

from stanza.models.common.doc import Document
from stanza.utils.conll import CoNLL


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data" / "external"
    test_data_destination = repo_root / "data" / "processed"

    for tb_dir in data_root.iterdir():
        if not tb_dir.is_dir():
            continue
        if len(tb_dir.name) != 3:
            continue

        lang_name = tb_dir.name
        print(f"\n{lang_name}:")

        splits_map = {
            "train": list(tb_dir.rglob("*train*.conllu")),
            "dev": list(tb_dir.rglob("*dev*.conllu")),
            "test": list(tb_dir.rglob("*test*.conllu")),
        }

        test_splits = splits_map["test"]
        if not test_splits:
            print(f"Found no test split for {lang_name}.")

        # Sort the sentences by length
        test_doc = CoNLL.conll2doc(test_splits[0])
        test_doc.sentences.sort(key=lambda s: len(s.text))

        # Divide the list of sentences up into parts of 20 and
        # take the middle 10 cutting points for the data set.
        doc_len = len(test_doc.sentences)
        dist = doc_len // 20
        subset = [test_doc.sentences[(i + 6) * dist] for i in range(10)]

        for sent in subset:
            sent_path = (
                test_data_destination / lang_name / f"{sent.sent_id}-original.conllu"
            )
            doc = Document([])
            doc.sentences.append(sent)
            sent_path.parent.mkdir(parents=True, exist_ok=True)
            CoNLL.write_doc2conll(doc, sent_path)
            print(sent.text)


if __name__ == "__main__":
    main()
