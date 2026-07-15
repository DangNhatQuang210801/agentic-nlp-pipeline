# Agentic NLP Pipeline

A multilingual project testing whether a dependency parsing agent can improve CoNLL-U dependency annotation quality compared to direct LLM prompting.

## Languages and Data Sources

- eng: [UD_English-GUM](https://github.com/UniversalDependencies/UD_English-GUM.git)
- mar: [UD_Marathi-UFAL](https://github.com/UniversalDependencies/UD_Marathi-UFAL.git)
- nan: [UD_Taiwanese-Ckiplab](https://github.com/ckiplab/ud.git)
- nds: [UD_Low_Saxon-LSDC](https://github.com/UniversalDependencies/UD_Low_Saxon-LSDC.git)
- vie: [UD_Vietnamese-VTB](https://github.com/UniversalDependencies/UD_Vietnamese-VTB.git)

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
