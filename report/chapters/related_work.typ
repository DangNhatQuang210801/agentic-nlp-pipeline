#import "../template/lib.typ" as template

= Related Work

While all major Python NLP libraries supply dependency parser components, their approaches are quite different.

*SpaCy.* The standard DependencyParser component in SpaCy is transition-based and uses a variation of the non-monotonic arc-edger transition system developed by #cite(<honnibal-johnson-2014-joint>, form: "prose"). It learns labelled dependency parsing jointly with sentence segmentation. Modifications are made to enable the generation of non-projective parses @spacy-dependencyparser. According to their official documentation, SpaCy currently achieves an unlabeled attachment score of over 95% on the Penn Treebank, an English-only treebank that is, however, not directly comparable to Universal Dependencies @spacy-facts-figures.

*Stanza.* The Stanza dependency parser component is based on a bidirectional LSTM architecture. It uses pretrained word embeddings, frequent word and lemma embeddings, character-level word embeddings, summed XPOS and UPOS embeddings, and summed FEATS embeddings as inputs. The average UAS across treebanks is reported to be at 77%. Considering only high-resource languages, this average rises to about 87%, while for low-resource languages it can be estimated to be no better than 70% (LAS of 63% plus a generous epsilon) @qi-etal-2018-universal. #cite(<nguyen-2020-implementing>, form: "prose") trained a dependency parser for Vietnamese following a very similar architecture and measured a UAS of 77%, which is in good alignment with the above figures.

*NLTK.* The NLTK has different dependency parsers available, but it is not easy to find out a whole lot of details about them. Judging from the NLTK-wiki it seems that—compared to SpaCy and Stanza—their approach is somewhat outdated @nltk-dependency-parsing-wiki.

None of the above tools use generative LLMs directly, but a lot of current research is aimed at figuring out how to leverage the power of LLMs to push the limits of dependency parsing accuracy. Due to their cross-lingual transfer capabilities, these new approaches promise to be especially valuable in the low-resource domain as demonstrated by #cite(<liu-etal-2026>, form: "prose") using a self-optimization algorithm. Their study finds that depending on the low-resource language, models in the 7B to 8B range achieve average unlabeled attachment scores of 65% to 83%. Most interestingly, using Qwen2.5-7B-instruct they report a UAS of around 75% for Vietnamese, which is on-par (but notably not better than) the more traditional BiLSTM-based tools.

Further benchmarking results of different models and systems can be found at #link("nlpprogress.com/english/dependency_parsing.html")[nlpprogress.com].
