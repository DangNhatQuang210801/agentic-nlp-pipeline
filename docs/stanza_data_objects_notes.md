# Stanza Data Objects Notes

Use Stanza objects internally. Convert to dictionaries only at JSON, file, or LLM-tool boundaries.

## Main Types

- `Document`: full annotated document.
  - Key property: `document.sentences`
  - Useful method: `document.to_dict()`
- `Sentence`: one sentence.
  - Key properties: `sentence.words`, `sentence.tokens`, `sentence.sent_id`, `sentence.text`, `sentence.dependencies`
  - Useful method: `sentence.to_dict()`
- `Word`: syntactic word used by POS, lemma, and dependency parsing.
  - Key properties: `word.id`, `word.text`, `word.lemma`, `word.upos`, `word.xpos`, `word.feats`, `word.head`, `word.deprel`, `word.deps`, `word.misc`
  - Useful method: `word.to_dict()`

## CoNLL-U

Read CoNLL-U with:

```python
from stanza.utils.conll import CoNLL

doc = CoNLL.conll2doc(input_file="path/to/file.conllu")
```

When constructing Stanza objects by hand, prefer Stanza's own field constants:

```python
from stanza.models.common.doc import ID, TEXT, UPOS, Sentence

sent = Sentence([
    {ID: 1, TEXT: "I", UPOS: "PRON"},
    {ID: 2, TEXT: "like", UPOS: "VERB"},
])
```

Write a `Document` back to CoNLL-U with:

```python
CoNLL.write_doc2conll(doc, "path/to/output.conllu")
```

Stanza maps CoNLL-U fields to internal lowercase property names:

```text
ID -> id
FORM -> text
LEMMA -> lemma
UPOS -> upos
XPOS -> xpos
FEATS -> feats
HEAD -> head
DEPREL -> deprel
DEPS -> deps
MISC -> misc
```

## Project Rule

For this repo:

```python
# Good internal shape
Document -> Sentence -> Word

# Boundary only
word.to_dict()
sentence.to_dict()
document.to_dict()
```

Avoid custom mirror dataclasses for CoNLL-U rows unless they add behavior Stanza does not already provide.

## Local References

- Stanza source checkout: `D:\stanza`
- Data objects source: `D:\stanza\stanza\models\common\doc.py`
- CoNLL conversion source: `D:\stanza\stanza\utils\conll.py`
- Official docs: https://stanfordnlp.github.io/stanza/data_objects.html
