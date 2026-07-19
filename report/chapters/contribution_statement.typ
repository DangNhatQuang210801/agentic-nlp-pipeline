#import "../template/lib.typ" as template

= Author Contribution Statements

In this section, each author describes their contribution in their own words.

*Dang Nhat Quang.* Initially, I did Vietnamese treebank inspection and output verification. I did most of the work around design, implementation, testing, and documentation of the shared tools: protocol, morphology lookup, n-gram retrieval, and bag-of-words retrieval. I wrote the section about "Our Approach".

Disclosure: GitHub Copilot was used only for minor code cleanup. All changes were reviewed and tested by the authors. References used involve codes such as Stanza examples and functions structure designs.

*Han Dai.* I assisted in the initial data-fetching phase of the project. I wrote and ran the scripts to download the UD treebanks, verified their train/dev/test splits, and performed the preliminary analysis on this data. I supported the creation of the first draft of the project presentation, laying out the initial structure and content that the team later refined.

AI usage note: I ran scripts in Google Colab and accepted the AI recommendations to format the output in a more readable layout.

*Nikhila Gadge.* I assisted in bringing the Taiwanese data into UD format, and uploaded fetching data file in the pipeline. Also assisted in running the pipeline with tools for English, and Lower Saxon using Qwen HF model using Colab. I wrote the "Expermiment" section of this report and helped create the presentation slides.

*Jan Raring.* With respect to the code, I wrote the agent harness, the tree validation tool and most of the code around the experiments. I wrote the following sections of this report: Abstract, Introduction, Related Work, Results, and Conclusion. Also, I took on the role of managing the project and keeping the oversight.

I used the some AI-assistance for creating a first project roadmap, for helping me find some bugs and for the refinement of the system prompts.

