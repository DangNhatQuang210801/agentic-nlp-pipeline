#import "../template/lib.typ" as template

= Introduction

In natural language processing (NLP), dependency parsing refers to the task of constructing sentence-level dependency trees. Syntactically, a dependency tree is a directed, connected, acyclic graphs with words as nodes and _dependency-head_-relations as edges. Semantically, it reflects a sentence's dependency structure: Each word is linked to the head of the phrase that it is part of. There is always exactly one word that does not itself have a head. This word is called the _root_ of the sentence. For additional specification, the edges of such a dependency tree are often annotated with a label taken from a pre-defined set of possible dependency relations.

Large collections of such dependency trees—so called _treebanks_—are being used in a number of linguistic subfields: In syntax they shed light on the possible sentence structures a language allows for; in grammar they help explain to what sort of agreement constraints words in a phrase are subjected to; and in linguistic typology, languages are classified and compared based on the prevailing patterns in word order.

One popular guideline for dependency annotation is the _Universal Dependencies_ (UD) framework along the its CoNLL-U format.
//treebansk, ud, ...
//columns of CoNLL-U
//how many languages are covered by UD?

One way or the other, the creation of a new treebank typically requires a lot of manual labor. To reduce the work load as much as possible, many projects opt for a semi-automated approach, using a pre-trained tagger for intermediate annotations that, in a second step, get corrected by a human. For high-resource languages, taggers are available via all major NLP libraries (Stanza, SpaCy, NLTK), but for the majority of the worlds roughly 7000 languages no such tools exist.

As the training of a tagger itself postulates the existence of a treebank to serve as training data, the problem of not having a pre-trained model is not one to be resolved easily. Additional challenges stem from the fact that many low resource languages have not undergone standardization, and so the little amount of data that is available displays high entropy. This throws annotators back into the situation of having to do everything by hand, which—depending on the availability of funding—can make the creation of a new treebank simply unaffordable.

// This observation has instigated us to test whether an AI agent, i.e. an LLM equipped with adequate tools, can function as a general purpose dependency parser, thereby eliminating the need to train purpose-built neural networks as used by the aforementioned libraries.
