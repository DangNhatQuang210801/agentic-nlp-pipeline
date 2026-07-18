#import "../template/lib.typ" as template

= Our Approach

== Agentic Tools

The implemented tools cover morphology lookup, surface form and part of speech retrieval, bag of words retrieval, and dependency tree validation. Retrieved examples include complete token annotations with heads and dependency relations. The tools are designed in a way so that they can be applied to Vietnamese, German, Marathi, and Low Saxon treebanks (or any other one as long as they follow the CoNLL-U format). Their common interfaces support consistent evidence retrieval and structural checks under the same experimental conditions. Stanza Document and Sentence objects are used throughout the tool layer. CoNLL-U files are read through the Stanza CoNLL reader, and every tool follows a shared JSON interface. This design avoids separate implementations for individual languages.

*Statistical Morphology Lookup Tool.* The tool creates a case insensitive index from each training treebank. For every input token, it returns observed LEMMA, UPOS, and FEATS candidates with frequency counts. The result provides statistical, language specific morphological evidence through one common interface.

*Surface Form and Part of Speech Retrieval Tool.* This method compares a query sentence with training sentences through surface form and UPOS sequence overlap. It returns the highest scoring examples with FORM, LEMMA, UPOS, FEATS, HEAD, and DEPREL annotations.

*Bag of Words Retrieval Tool.* This method combines multiset word overlap with sentence length similarity. It provides a complementary baseline for example selection and returns complete annotations, including dependency heads and relations.

*Dependency Tree Validation Tool.* The tool checks whether head indices are valid, whether each tree contains exactly one root, and whether the dependency structure is acyclic. Invalid structures can therefore be identified before evaluation.

