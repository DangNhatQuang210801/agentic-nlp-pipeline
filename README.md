# Agentic NLP Pipeline

A multilingual project testing whether a dependency parsing agent can improve CoNLL-U dependency annotation quality compared to direct LLM prompting.

Team project for the course AI Engineering (Linguistic Data Science Lab, Ruhr University Bochum), submitted in July 2026. The full write-up is in [`report/Report.pdf`](report/Report.pdf).


## Languages and Data Sources

We have a special interest in low-resource languages and chose to run our experiment on the native languages of our team members.

- eng: [UD_English-GUM](https://github.com/UniversalDependencies/UD_English-GUM.git)
- mar: [UD_Marathi-UFAL](https://github.com/UniversalDependencies/UD_Marathi-UFAL.git)
- nan: [UD_Taiwanese-Ckiplab](https://github.com/ckiplab/ud.git) (Ckiplab's UD conversion of the Sinica Treebank, which is Mandarin text from Taiwan; the report calls it Taiwanese)
- nds: [UD_Low_Saxon-LSDC](https://github.com/UniversalDependencies/UD_Low_Saxon-LSDC.git)
- vie: [UD_Vietnamese-VTB](https://github.com/UniversalDependencies/UD_Vietnamese-VTB.git)


## Experiment

In our study, the following sources of truth are compared against each other:
- Gold Universal Dependencies treebanks as reference data
- Direct LLM prompting without agentic repair
- Agentic NLP pipeline with iterative inspection and correction


## Model and Tools

Both settings use Qwen3.5-9B: a GGUF build (Q4_K_M) served by llama.cpp through its OpenAI-compatible API (`agentic_nlp_pipeline/models/llama_cpp.py`), and a 4-bit NF4 build loaded with transformers (`agentic_nlp_pipeline/models/local.py`). The experiment parses 50 sentences, 10 per language.

In the agentic setting the model can call these tools (`agentic_nlp_pipeline/tools/`):

- `morphology_lookup.py`: returns frequency-ranked lemma, UPOS and FEATS candidates observed for each word form in the treebank.
- `knn_retrieval.py`: retrieves annotated sentences by n-gram overlap of word forms and UPOS tags.
- `bow_retrieval.py`: retrieves annotated sentences by word overlap and similar sentence length.
- `tree_validation.py`: checks that a predicted parse is a connected, acyclic tree with valid head ids.
- `base.py`: the shared tool protocol the agent harness uses to call the tools.


## Evaluation Metrics

The scope of our project is reduced to predicting the HEAD attribute. The most important evaluation metric is therefore the unlabeled attachment score (UAS), which is just the ratio of correctly predicted heads.


## Reproduction
**Requirements:**
- Git
- Python 3.12
- Poetry
- (Optional) A local chat completion server (e.g. llama.cpp) available at http://localhost:8080/v1

Clone the repository with
```shell
git clone git@github.com:DangNhatQuang210801/agentic-nlp-pipeline.git
```

Open the repository root in the terminal and install the project dependencies with
```shell
poetry install
```

> [!WARNING]  
> Due to what hardware was available to us, this installs an XPU-version of PyTorch. If you want to run the experiment on Cuda, replace this with the normal version. PyTorch is not needed at all if you run the experiment using a local llama.cpp server for inference.

To fetch the required third-party data, run
```shell
poetry run python scripts/fetch_corpus_data.py
```

After that, run the following script to sample 10 sentences per language with increasing length:
```shell
poetry run python scripts/produce_datasets.py
```

To parse the dependencies of the sample sentences without tools run
```shell
poetry run python scripts/experiment_without_tools.py
```

To parse the dependencies of the sample sentences with tools run
```shell
poetry run python scripts/experiment_with_tools.py
```

> [!WARNING]  
> The current setup expects a chat completion server to be running at http://localhost:8080/v1. Use `LocalModel` if you want to load a model in PyTorch instead.

Once the experiments are done running, you can compile the results into a CSV-file by executing
```shell
poetry run python scripts/data_analysis.py
```


## Results

From the report (`report/chapters/results.typ`, data in `data/processed/parse.csv`):

| Language | UAS without tools | UAS with tools | Change |
|---|---|---|---|
| English | 0.188 | 0.336 | +78.3% |
| Marathi | 0.554 | 0.557 | +0.5% |
| Taiwanese (Mandarin) | 0.783 | 0.682 | -13.0% |
| Low Saxon | 0.228 | 0.313 | +37.0% |
| Vietnamese | 0.524 | 0.550 | +5.0% |

- With tools, the average number of generated tokens fell from 7,801 to 4,980 (about 36% fewer), and far more runs produced a final answer within the token limit, especially for English and Low Saxon.
- UAS improved clearly only for English and Low Saxon; it fell for Taiwanese and stayed roughly level for Marathi and Vietnamese.
- With tools, the model produced fewer formally valid trees on longer sentences, even though a tree validation tool was available.
- The sample is small (10 sentences per language), so these differences are indicative, not conclusive.


## Team

Contributions as stated by each author in `report/chapters/contribution_statement.typ`:

- **Dang Nhat Quang:** inspection of the Vietnamese treebank and output verification; most of the design, implementation, testing and documentation of the shared tools (protocol, morphology lookup, n-gram retrieval, bag-of-words retrieval); the "Our Approach" section of the report.
- **Han Dai:** download scripts for the UD treebanks, checks of their splits and the preliminary data analysis; the first draft of the presentation.
- **Nikhila Gadge:** bringing the Taiwanese data into UD format, the data-fetching file, runs with tools for English and Low Saxon on Colab; the "Experiment" section of the report and the slides.
- **Jan Raring:** the agent harness, the tree validation tool and most of the experiment code; the Abstract, Introduction, Related Work, Results and Conclusion sections; project management.
