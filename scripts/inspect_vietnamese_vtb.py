from pathlib import Path
from collections import Counter


FIELDS = ("ID", "FORM", "LEMMA", "UPOS", "XPOS", "FEATS", "HEAD", "DEPREL", "DEPS", "MISC")
TARGET = ("FORM", "LEMMA", "UPOS", "FEATS", "HEAD", "DEPREL")


def read_conllu(path):
    sent = {"meta": {}, "tokens": []}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            if sent["tokens"]:
                yield sent
            sent = {"meta": {}, "tokens": []}
        elif line.startswith("#"):
            if "=" in line:
                key, value = line[1:].split("=", 1)
                sent["meta"][key.strip()] = value.strip()
        else:
            cols = line.split("\t")
            if len(cols) != 10:
                continue
            if "-" in cols[0] or "." in cols[0]:
                continue
            sent["tokens"].append(dict(zip(FIELDS, cols)))
    if sent["tokens"]:
        yield sent


def inspect_file(path):
    summary = {
        "sentences": 0,
        "tokens": 0,
        "non_empty": Counter(),
        "forms_with_spaces": 0,
        "examples": [],
    }
    for sent in read_conllu(path):
        summary["sentences"] += 1
        summary["tokens"] += len(sent["tokens"])
        if len(summary["examples"]) < 5:
            summary["examples"].append(sent)
        for token in sent["tokens"]:
            if " " in token["FORM"]:
                summary["forms_with_spaces"] += 1
            for field in FIELDS:
                if token[field] != "_":
                    summary["non_empty"][field] += 1
    return summary


def main():
    repo = Path(__file__).resolve().parents[1] / "data" / "vietnamese" / "raw" / "UD_Vietnamese-VTB"
    files = sorted(repo.rglob("*.conllu"))

    print("CoNLL-U files:")
    for path in files:
        print(f"- {path.relative_to(repo)}")

    print("\nDataset summary:")
    for path in files:
        summary = inspect_file(path)
        available = ", ".join(
            f"{field}={'yes' if summary['non_empty'][field] else 'column only'}"
            for field in TARGET
        )
        print(f"- {path.relative_to(repo)}: {summary['sentences']} sentences, {summary['tokens']} tokens")
        print(f"  {available}")
        print(f"  FORM values containing spaces: {summary['forms_with_spaces']}")

    test = repo / "vi_vtb-ud-test.conllu"
    print("\nFive test examples:")
    for sent in inspect_file(test)["examples"]:
        print(f"\n{sent['meta'].get('sent_id', '?')}: {sent['meta'].get('text', '')}")
        for token in sent["tokens"]:
            parts = [token[field] for field in TARGET]
            print("  " + "\t".join(parts))


if __name__ == "__main__":
    main()
