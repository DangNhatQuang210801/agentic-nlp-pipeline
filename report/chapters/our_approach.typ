#import "../template/lib.typ" as template

= Our Approach

In this study, an agent is defined as a system that combines a language model with callable external tools and can select relevant tools during dependency parsing. The underlying principle is that low-resource languages are not necessarily resource-free. Dictionaries, grammars, partial annotations, and small treebanks may still provide useful linguistic evidence. Access to these resources can support more informed predictions, resembling how a human analyst consults reference materials when examining an unfamiliar language.

In this study we use treebanks as a source of supplementary information and although the project addresses low-resource settings, the use of treebanks as reference sources does not assume that complete dependency annotations are generally available. Dependency parsing is commonly performed after tokenization, lemmatization, part-of-speech tagging, and morphological analysis. UPOS and FEATS may therefore be available before dependency structures are assigned. These intermediate annotations can support the morphology and retrieval tools. Sentence retrieval can also benefit from any amount of verified parsing data. A small seed set of completed analyses may already provide relevant examples for sentences with similar lexical or syntactic patterns. The training treebanks are therefore used as bounded sources of linguistic evidence and as controlled reference data for evaluation, not as an assumption that all target sentences are already annotated.

== Agentic Tools

The project uses a shared dependency parsing pipeline to compare direct parsing with tool-supported parsing across multiple languages. Sentences are represented as JSON-style lists of token dictionaries at the pipeline boundary and as Stanza Document and Sentence objects internally. This keeps CoNLL-U input, tool arguments, and returned annotations consistent across languages.

Four reusable tools provide linguistic evidence and structural checks. The tools are designed in a way so that they can be applied to Vietnamese, German, Taiwanese, Marathi, and Low Saxon, or really all other UD treebanks. This design avoids separate implementations for individual languages.

*Statistical Morphology Lookup Tool.* The tool creates a case insensitive index from each training treebank. For every input token, it returns observed LEMMA, UPOS, and FEATS candidates with frequency counts. The result provides statistical, language specific morphological evidence through one common interface.

*Surface Form and Part of Speech Retrieval Tool.* This method compares a query sentence with training sentences through surface form and UPOS sequence overlap. It returns the highest scoring examples with FORM, LEMMA, UPOS, FEATS, HEAD, and DEPREL annotations.

*Bag of Words Retrieval Tool.* This method combines multiset word overlap with sentence length similarity. It provides a complementary baseline for example selection and returns complete annotations, including dependency heads and relations.

*Dependency Tree Validation Tool.* The tool checks whether head indices are valid, whether each tree contains exactly one root, and whether the dependency structure is acyclic. Invalid structures can therefore be identified before evaluation.
