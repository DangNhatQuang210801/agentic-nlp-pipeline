import subprocess
from pathlib import Path

# from sklearn.model_selection import train_test_split
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


## Download datasets

repos = {
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

repo_root = Path(__file__).resolve().parents[1]
data_dir = repo_root / "data" / "external"

for lang, (name, repo) in repos.items():
    print(f"Fetching {name}:")
    repo_dir = data_dir / lang / name  # original repo_dir = Path(f"data/{name}")
    if not repo_dir.exists():
        subprocess.run(["git", "clone", repo, str(repo_dir)])
    else:
        print(f"good to go: {repo_dir}")
    print()


## Glance at data

treebanks_root = data_dir


for tb_dir in treebanks_root.iterdir():
    if not tb_dir.is_dir():
        continue

    repo_name = tb_dir.name
    print(f"\n{repo_name}:")

    splits_map = {
        "train": list(tb_dir.rglob("*train*.conllu")),
        "dev": list(tb_dir.rglob("*dev*.conllu")),
        "test": list(tb_dir.rglob("*test*.conllu")),
    }

    for split_name, files_found in splits_map.items():
        if files_found:
            for fp in files_found:
                sns = read_sns(fp)
                doc = CoNLL.conll2doc(input_file=str(fp))  # added token counts
                token_count = sum(
                    len(sent.words) for sent in doc.sentences
                )  # added token counts
                print(f"  {fp.name}: {len(sns)} sentences")
                print(f"  {fp.name}: {token_count} tokens")
        else:
            print(f"  {repo_name}: 0 sentences")
            print(f"  {repo_name}: 0 tokens")
