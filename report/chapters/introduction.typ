#import "../template/lib.typ" as template

= Introduction

// Expanding on the abstract:
In natural language processing (NLP), dependency parsing refers to the task of constructing sentence-level dependency trees. Syntactically, a dependency tree is a directed, connected, acyclic graphs with words as nodes and _dependency-head_-relations as edges. Semantically, it reflects a sentence's dependency structure: Each word is linked to the head of the phrase it is part of. There is always exactly one word that does not itself have a head. This word is called the _root_ of the sentence. The edges of such a dependency tree are often annotated with a label taken from a pre-defined set of possible dependency relations.


// Large collections of such dependency trees—so called treebanks—are a valuable resource in the study of syntax, grammar, and typology.
//treebansk, ud, ...

// The manual creation of such dependency trees, however, is a very laborious undertaking which should, if possible, be performed at least semi-automatically.
// For high-resource languages, many popular NLP libraries (Stanza, NLTK, SpaCy) provide the necessary tooling for exactly this sort of (pre-)annotation.
// In the case of low-resource languages, however, such tools are typically not available.
// This observation has instigated us to test whether an AI agent, i.e. an LLM equipped with adequate tools, can function as a general purpose dependency parser, thereby eliminating the need to train purpose-built neural networks as used by the aforementioned libraries.
