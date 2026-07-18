import subprocess
from pathlib import Path

from sklearn.model_selection import train_test_split
from stanza.utils.conll import CoNLL


def read_sns(fp):
    try:
        with open(fp, "r", encoding="utf-8") as f:
            content = f.read()
        sns = [s.strip() for s in content.split("\n\n") if s.strip()]
        return sns
    except FileNotFoundError:
        return []


def write_sns(fp, sns):
    with open(fp, "w", encoding="utf-8") as f:
        for s in sns:
            f.write(s)
            f.write("\n\n")


REPOS = {
    "eng": (
        "UD_English-GUM",
        "https://github.com/UniversalDependencies/UD_English-GUM.git",
    ),
    "mar": (
        "UD_Marathi-UFAL",
        "https://github.com/UniversalDependencies/UD_Marathi-UFAL.git",
    ),
    "vie": (
        "UD_Vietnamese-VTB",
        "https://github.com/UniversalDependencies/UD_Vietnamese-VTB.git",
    ),
    "nds": (
        "UD_Low_Saxon-LSDC",
        "https://github.com/UniversalDependencies/UD_Low_Saxon-LSDC.git",
    ),
    "nan": ("UD_Taiwanese-Ckiplab", "https://github.com/ckiplab/ud.git"),
}


def main():
    # Download datasets
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "external"

    for lang, (name, repo) in REPOS.items():
        print(f"Fetching {name}:")
        repo_dir = data_dir / lang / name  # original repo_dir = Path(f"data/{name}")
        if not repo_dir.exists():
            subprocess.run(["git", "clone", repo, str(repo_dir)])
        else:
            print(f"good to go: {repo_dir}")
        print()

    # Extra work for Taiwanese
    prep_taiwanese_data(repo_root)

    # Glance at data
    for tb_dir in data_dir.iterdir():
        if not tb_dir.is_dir():
            continue

        repo_name = tb_dir.name
        print(f"\n{repo_name}:")

        splits_map = {
            "train": list(tb_dir.rglob("*train*.conllu")),
            "dev": list(tb_dir.rglob("*dev*.conllu")),
            "test": list(tb_dir.rglob("*test*.conllu")),
        }

        for files_found in splits_map.values():
            if files_found:
                for fp in files_found:
                    sns = read_sns(fp)
                    doc = CoNLL.conll2doc(input_file=str(fp))
                    token_count = sum(len(sent.words) for sent in doc.sentences)
                    print(f"  {fp.name}: {len(sns)} sentences")
                    print(f"  {fp.name}: {token_count} tokens")
            else:
                print(f"  {repo_name}: 0 sentences")
                print(f"  {repo_name}: 0 tokens")


def prep_taiwanese_data(repo_root: Path):
    input_file = (
        repo_root
        / "data"
        / "external"
        / "nan"
        / "UD_Taiwanese-Ckiplab"
        / "data"
        / "sinica_ud.conllu"
    )
    output_file = (
        repo_root
        / "data"
        / "external"
        / "nan"
        / "UD_Taiwanese-Ckiplab"
        / "sinica_ud_converted.conllu"
    )

    # converting the input file to the Conllu format:
    with (
        input_file.open("r", encoding="utf-8") as f_in,
        output_file.open("w", encoding="utf-8") as f_out,
    ):
        for line in f_in:
            # Preserve comments and blank lines
            if line.startswith("#") or not line.strip():
                f_out.write(line)
                continue

            cols = line.rstrip("\n").split("\t")

            # Skip malformed rows
            if len(cols) != 12:
                print(f"Skipping malformed row ({len(cols)} cols): {cols}")
                continue

            (
                tok_id,  # 1
                form,  # 2
                lemma,  # 3
                coarse_pos,  # 4 (unused)
                xpos,  # 5
                feats,  # 6
                upos,  # 7
                head,  # 8
                empty,  # 9 (unused)
                deps,  # 10
                deprel,  # 11
                misc,  # 12
            ) = cols

            conllu = [
                tok_id,  # ID
                form,  # FORM
                lemma,  # LEMMA
                upos,  # UPOS
                xpos,  # XPOS
                feats,  # FEATS
                head,  # HEAD
                deprel,  # DEPREL
                deps,  # DEPS
                misc,  # MISC
            ]

            f_out.write("\t".join(conllu) + "\n")

    # Insert text metadata
    doc = CoNLL.conll2doc(output_file)
    for sent in doc.sentences:
        forms = [w.text for w in sent.words]
        sent.text = "".join(forms)
        sent.add_comment(f"text = {sent.text}")
    CoNLL.write_doc2conll(doc, output_file)

    """
    UD_Taiwanese-Ckiplab original data doesn't have train, dev, test, so I:
    1.  split `sinica_ud.conllu` to train 80% & test 20%
    2.  rename the file to match the other UD langs datasets
    3.  take 10% from train as dev
    """

    path = Path("data/external/nan/UD_Taiwanese-Ckiplab/")

    input_file = Path(path, "sinica_ud_converted.conllu")

    if input_file.exists():
        sentences = read_sns(input_file)

        if len(sentences) > 1:
            twsns_train, twsns_test = train_test_split(
                sentences,
                test_size=0.2,
                random_state=42,  # 20% test, 80% train
            )

            train_file = input_file.with_name(f"{input_file.stem}-train.conllu")
            test_file = input_file.with_name(f"{input_file.stem}-test.conllu")

            write_sns(train_file, twsns_train)
            write_sns(test_file, twsns_test)

        else:
            print("Not enough sentences to split.")
    else:
        print(f"{input_file} not found.")


if __name__ == "__main__":
    main()
