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
    # "nan": ("UD_Taiwanese-Ckiplab", "https://github.com/ckiplab/ud.git"),
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


## Extra works on TW data
"""
UD_Taiwanese-Ckiplab original data doesn't have train, dev, test, so I:
1.  split `sinica_ud.conllu` to train 80% & test 20%
2.  rename the file to match the other UD langs datasets
3.  take 10% from train as dev
"""

"""
tw_path = Path("data/externaml/nan/UD_Taiwanese-Ckiplab/data/sinica_ud.conllu")
tw_path_train = Path(
    "data/externaml/nan/UD_Taiwanese-Ckiplab/data/sinica_ud-train.conllu"
)
test_temp = Path("data/externaml/nan/UD_Taiwanese-Ckiplab/data/sinica_ud-test.conllu")

if tw_path.exists():
    tw_allsns = read_sns(tw_path)
    print(f"Total sentences: {len(tw_allsns)}")

    if len(tw_allsns) > 1:
        twsns_train, twsns_test = train_test_split(
            tw_allsns, test_size=0.2, random_state=42
        )

        print(f"{len(twsns_train)} sentences to Training")
        write_sns(tw_path_train, twsns_train)

        print(f"{len(twsns_test)} sentences to Testing")
        write_sns(test_temp, twsns_test)
    else:
        print("Not enough sentences to split")


ud_path_train = Path(
    "data/externaml/nan/UD_Taiwanese-Ckiplab/data/sinica_ud-train.conllu"
)
ud_path_test = Path(
    "data/externaml/nan/UD_Taiwanese-Ckiplab/data/sinica_ud-test.conllu"
)

new_train = Path(
    "data/externaml/nan/UD_Taiwanese-Ckiplab/data/zh_ckiplab-ud-train.conllu"
)
new_test = Path(
    "data/externaml/nan/UD_Taiwanese-Ckiplab/data/zh_ckiplab-ud-test.conllu"
)

if ud_path_train.exists():
    ud_path_train.rename(new_train)
    print(f"Renamed '{ud_path_train.name}' > '{new_train.name}'")
else:
    print(f"not found")

if ud_path_test.exists():
    ud_path_test.rename(new_test)
    print(f"Renamed '{ud_path_test.name}' > '{new_test.name}'")
else:
    print(f"not found")


train_tw = Path(
    "data/externaml/nan/UD_Taiwanese-Ckiplab/data/zh_ckiplab-ud-train.conllu"
)
train_tw_new = Path(
    "data/externaml/nan/UD_Taiwanese-Ckiplab/data/zh_ckiplab-ud-train-new.conllu"
)
train_tw_dev = Path(
    "data/externaml/nan/UD_Taiwanese-Ckiplab/data/zh_ckiplab-ud-dev.conllu"
)

if train_tw.exists():
    all_train_sns = read_sns(train_tw)
    print(f"Total training sentences: {len(all_train_sns)}")

    if len(all_train_sns) > 1:
        new_train_sns, dev_sns = train_test_split(
            all_train_sns, test_size=0.1, random_state=42
        )

        print(f"{len(new_train_sns)} to New Training")
        write_sns(train_tw_new, new_train_sns)

        print(f"{len(dev_sns)} sentences to Development")
        write_sns(train_tw_dev, dev_sns)

    else:
        print("Not enough sentences")
"""


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
