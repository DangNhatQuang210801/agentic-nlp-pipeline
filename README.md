# Agentic NLP Pipeline

A multilingual project testing whether a dependency parsing agent can improve CoNLL-U dependency annotation quality compared to direct LLM prompting. 


## Languages and Data Sources

We have a special interest in low-resource languages and chose to run our experiment on the native languages of our team members.

- eng: [UD_English-GUM](https://github.com/UniversalDependencies/UD_English-GUM.git)
- mar: [UD_Marathi-UFAL](https://github.com/UniversalDependencies/UD_Marathi-UFAL.git)
- nan: [UD_Taiwanese-Ckiplab](https://github.com/ckiplab/ud.git)
- nds: [UD_Low_Saxon-LSDC](https://github.com/UniversalDependencies/UD_Low_Saxon-LSDC.git)
- vie: [UD_Vietnamese-VTB](https://github.com/UniversalDependencies/UD_Vietnamese-VTB.git)


## Experiment

In our study, the following sources of truth are compared against each other:
- Gold Universal Dependencies treebanks as reference data
- Direct LLM prompting without agentic repair
- Agentic NLP pipeline with iterative inspection and correction


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
