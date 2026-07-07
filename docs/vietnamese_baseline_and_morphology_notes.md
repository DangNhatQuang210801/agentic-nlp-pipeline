# Vietnamese Baseline and Morphology Notes

Task note: Vietnamese-specific preparation for `1.3a-VI` and `2.5-VI`. GenAI was used to help inspect the repository, run the checks, and draft this note.

## 1.3a-VI — Vietnamese Baseline Preparation

Shared task `1.3a` is to train and evaluate a supervised parser on Vietnamese, but the Google Doc notes that tasks `1.0` to `1.3` should only proceed if there is time and `1.3a` depends on `1.0`.

Current safe scope:

- Do not train a parser yet.
- Use only the official UD_Vietnamese-VTB splits:
  - train: `data/vietnamese/raw/UD_Vietnamese-VTB/vi_vtb-ud-train.conllu`
  - dev: `data/vietnamese/raw/UD_Vietnamese-VTB/vi_vtb-ud-dev.conllu`
  - test: `data/vietnamese/raw/UD_Vietnamese-VTB/vi_vtb-ud-test.conllu`
- Keep `not-to-release/` out of baseline evaluation unless the team explicitly decides otherwise.
- Require model output to preserve the gold token IDs and `FORM` values before scoring dependencies.

Baseline readiness status:

| Item | Status |
|---|---|
| Official train/dev/test paths known | Ready |
| Sentence/token counts checked | Ready |
| CoNLL-U fields checked | Ready |
| Poetry environment installed | Ready |
| Stanza installed through Poetry | Ready |
| Shared Stanza train/eval method from task `1.0` | Waiting |
| Evaluation harness from task `0.4` | Waiting |

Recommended first Vietnamese baseline target:

1. Wait for task `1.0` training/evaluation method.
2. Run it on the official Vietnamese train/dev/test files.
3. Report UAS, LAS, and CLAS.
4. Add a short warning beside the scores if token preservation was not enforced.

## 2.5-VI — Vietnamese Morphology / Feature Lookup Check

Shared task `2.5` proposes a morphological analyzer or a feature lookup from the train treebank. For Vietnamese VTB, a direct `FEATS` lookup is not useful because the train split has no populated morphological features.

Train split lookup check:

| Check | Result |
|---|---:|
| Train sentences | 1,400 |
| Train tokens | 20,215 |
| Unique casefolded `FORM` values | 3,398 |
| `FORM` values with multiple lemmas | 77 |
| `FORM` values with multiple UPOS tags | 323 |
| `FORM` values with multiple lemma or UPOS analyses | 349 |
| Non-empty `FEATS` values | 0 |

Observed `FEATS` distribution:

| FEATS | Count |
|---|---:|
| `_` | 20,215 |

Common ambiguous lookup examples:

| FORM | Count | Ambiguity |
|---|---:|---|
| `và` | 253 | `CCONJ` / `SCONJ` |
| `là` | 229 | `AUX` / `SCONJ` |
| `có` | 208 | `VERB` / `ADV` |
| `được` | 161 | `AUX` / `ADV` / `VERB` / `PART` |
| `đến` | 136 | `ADP` / `VERB` / `ADV` / `PART` |
| `cho` | 132 | `ADP` / `VERB` / `ADV` |

Recommendation:

- Do not build a Vietnamese `FEATS` tool from this treebank; it would only return `_`.
- If task `2.5` needs a Vietnamese tool, make it a train-set lexical hint lookup for `LEMMA` and `UPOS`, not morphology.
- Return ranked candidates with counts rather than a single label, because common Vietnamese forms are ambiguous.
- Treat this as a weak hint for the agent, not a hard validator.

Minimal tool shape later:

```python
lookup("được") -> {
    "lemma_candidates": [("được", 161)],
    "upos_candidates": [("AUX", 92), ("ADV", 58), ("VERB", 10), ("PART", 1)],
    "feats_candidates": [("_", 161)],
}
```

This is enough for the agent to use observed train-treebank evidence without pretending that Vietnamese VTB contains morphological feature annotations.
