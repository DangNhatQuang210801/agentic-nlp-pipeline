# Vietnamese Data Audit

Task note: Vietnamese subset audit for UD_Vietnamese-VTB. Counts come from `scripts/inspect_vietnamese_vtb.py`.

## Source

- Treebank: Universal Dependencies Vietnamese VTB
- Local path: `data/vietnamese/raw/UD_Vietnamese-VTB`
- Official split files:
  - `vi_vtb-ud-train.conllu`
  - `vi_vtb-ud-dev.conllu`
  - `vi_vtb-ud-test.conllu`

The repository also contains `not-to-release/` files. These should not be used as official train/dev/test splits unless the team explicitly decides otherwise.

## Official Split Counts

| Split | File | Sentences | Tokens | FORM values containing spaces |
|---|---|---:|---:|---:|
| Train | `vi_vtb-ud-train.conllu` | 1,400 | 20,215 | 4,535 |
| Dev | `vi_vtb-ud-dev.conllu` | 1,123 | 26,162 | 5,497 |
| Test | `vi_vtb-ud-test.conllu` | 800 | 11,692 | 2,079 |

## CoNLL-U Field Coverage

The files use the standard 10 CoNLL-U columns:

`ID`, `FORM`, `LEMMA`, `UPOS`, `XPOS`, `FEATS`, `HEAD`, `DEPREL`, `DEPS`, `MISC`

For the official train/dev/test files:

| Field | Status |
|---|---|
| `FORM` | Available |
| `LEMMA` | Available |
| `UPOS` | Available |
| `FEATS` | Column exists, but values are `_` |
| `HEAD` | Available |
| `DEPREL` | Available |
| `DEPS` | Column exists, but values are `_` |
| `MISC` | Column exists, but values are `_` |

## Tokenization Notes

Vietnamese tokenization is a real evaluation issue here. Many `FORM` values contain spaces, for example:

- `bắt chuyện`
- `giật mình`
- `danh sách`
- `tên tuổi`
- `địa chỉ`
- `điện thoại`

This means whitespace tokenization cannot be used as a reliable proxy for CoNLL-U tokens. Any evaluation involving generated CoNLL-U must first align predicted tokens to the gold CoNLL-U token sequence, or require the model/system to preserve the provided token IDs and `FORM` values.

## Recommendation

For the first Vietnamese experiments, evaluate:

- `LEMMA`
- `UPOS`
- token preservation / `FORM` alignment

Do not prioritize `FEATS` for Vietnamese VTB because the official files contain only `_` in that column.

Delay `HEAD` and `DEPREL` evaluation until token alignment and output validation are stable. Dependency metrics such as UAS/LAS are useful later, but they become noisy if the model changes token boundaries or emits malformed trees.

## Reproduction

Run:

```powershell
python -m poetry run python scripts\inspect_vietnamese_vtb.py
```
