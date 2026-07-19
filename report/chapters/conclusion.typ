#import "../template/lib.typ" as template

= Conclusion

In the beginning, four lower-resource languages covering different degrees of NLP coverage, along with English for comparison, were selected as a case study in LLM-assisted dependency parsing. To this end, ten sentences of varying length have been selected for every language and each sentence got parsed twice: once by an LLM without tools and once by an agentic system with four different tools. These tools allowed the model to retrieve similar sentences from the treebanks train-split, to look up morphlogical details, and to get feedback on the structural soundness of the trees it was creating.

To analyse the results, different metrics on the final output and the generation process itself have been computed. Even though the results do not really speak in favor if agentic dependency parsing, they still leave us with a few interesting observations.
1. On average, having tools at its disposal, the model used much fewer tokens to give a final output, thereby increasing the chance of producing any output at all.
2. The downside to the faster output generation seems to be increased sloppiness when it comes to generating structurally valid trees.
3. Considering English and Low Saxon, seemingly contradictory to the above observations, it looks like having the option to use auxiliary tools noticeably increased the accuracy on longer, more complicated sentences.

Due to the tight time constraints of this project it is well-imaginable that we have not hit the ceiling of this agentic dependnecy parsing framework. The tools and prompting technique employed here are rather naive and tap in to only one possible source of information (treebanks). Therefore, refining them could still be an interesting further direction of research.
