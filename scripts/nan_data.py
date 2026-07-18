import subprocess
from pathlib import Path
from turtle import pd
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from IPython import display
from stanza.utils.conll import CoNLL
from scripts.fetch_corpus_data import read_sns, write_sns


eng_test = CoNLL.conll2doc("data/external/eng/UD_English-GUM/en_gum-ud-train.conllu")

print(eng_test.sentences[0])

count = 0
with open(
    "data/external/nan/UD_Taiwanese-Ckiplab/data/sinica_ud.conllu",
    encoding="utf-8",
) as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue

        print(line.rstrip())
        count += 1
        if count == 5:
            break


input_file = Path("data/external/nan/UD_Taiwanese-Ckiplab/data/sinica_ud.conllu")
output_file = Path("data/external/nan/UD_Taiwanese-Ckiplab/sinica_ud_converted.conllu")

# getting count of columns in each row of the input:
num = [
    len(line.rstrip().split("\t"))
    for line in open(input_file, encoding="utf-8")
    if line.strip() and not line.startswith("#")
]

print(Counter(num))

# converting the input file to the Conllu format:

with input_file.open("r", encoding="utf-8") as fin, \
     output_file.open("w", encoding="utf-8") as fout:

    for line in fin:
        # Preserve comments and blank lines
        if line.startswith("#") or not line.strip():
            fout.write(line)
            continue

        cols = line.rstrip("\n").split("\t")

        # Skip malformed rows
        if len(cols) != 12:
            print(f"Skipping malformed row ({len(cols)} cols): {cols}")
            continue

        (
            tok_id,     # 1
            form,       # 2
            lemma,      # 3
            coarse_pos, # 4 (unused)
            xpos,       # 5
            feats,      # 6
            upos,       # 7
            head,       # 8
            empty,      # 9 (unused)
            deps,       # 10
            deprel,     #11
            misc,       #12
        ) = cols

        conllu = [
            tok_id,      # ID
            form,        # FORM
            lemma,       # LEMMA
            upos,        # UPOS
            xpos,        # XPOS
            feats,       # FEATS
            head,        # HEAD
            deprel,      # DEPREL
            deps,        # DEPS
            misc,        # MISC
        ]

        fout.write("\t".join(conllu) + "\n")


## Extra works on TW data
"""
UD_Taiwanese-Ckiplab original data doesn't have train, dev, test, so I:
1.  split `sinica_ud.conllu` to train 80% & test 20%
2.  rename the file to match the other UD langs datasets
3.  take 10% from train as dev
"""


path = Path("data/external/nan/UD_Taiwanese-Ckiplab/data/external/nan/UD_Taiwanese-Ckiplab/")

input_file = Path(path + "sinica_ud_converted.conllu")


if input_file.exists():
    sentences = read_sns(input_file)
    print(f"Total sentences: {len(sentences)}")

    if len(sentences) > 1:
        twsns_train, twsns_test = train_test_split(
            sentences, test_size=0.2, random_state=42
        )

        train_file = input_file.with_name(f"{input_file.stem}-train.conllu")
        test_file = input_file.with_name(f"{input_file.stem}-test.conllu")

        write_sns(train_file, twsns_train)
        write_sns(test_file, twsns_test)

        print(f"Training: {train_file}")
        print(f"Testing : {test_file}")

    else:
        print("Not enough sentences to split.")
else:
    print(f"{input_file} not found.")

