# Agentic NLP Pipeline

A multilingual project testing whether an agentic NLP pipeline can improve CoNLL-U annotation quality compared with traditional NLP tools and direct LLM prompting.

## Planned Languages

- Vietnamese: Universal Dependencies Vietnamese VTB
- German
- Marathi
- Low Saxon

## Baseline Systems

- Gold Universal Dependencies treebanks as reference data
- Traditional NLP tools for automatic CoNLL-U prediction
- Direct LLM prompting without agentic repair
- Agentic NLP pipeline with iterative inspection and correction

## Evaluation Metrics

Initial evaluation will focus on fields that can be compared directly across CoNLL-U files:

- FORM/tokenization agreement
- LEMMA accuracy
- UPOS accuracy
- FEATS accuracy when morphological features are available
- HEAD attachment accuracy, later
- DEPREL accuracy, later

The first stage is dataset verification: inspect available treebanks, confirm field coverage, and identify tokenization issues before running any parser, LLM prompt, or repair agent.
