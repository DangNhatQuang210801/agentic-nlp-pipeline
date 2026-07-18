from pathlib import Path

from stanza.utils.conll import CoNLL

from agentic_nlp_pipeline import DepParseAgent, LocalModel, Experiment
from agentic_nlp_pipeline.experiment import PARSED_AGENTICALLY
from agentic_nlp_pipeline.agentic import templates
from agentic_nlp_pipeline.tools import (
    BagOfWordsRetrievalTool,
    KNNRetrievalTool,
    StatisticalMorphologyLookupTool,
    TreeValidationTool,
)


def main():
    # Find root dir
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data" / "processed" / "nds"

    # Set up agent
    model = LocalModel(model_id = "Qwen/Qwen3.5-9B", gguf_file=None, enable_thinking=True)
    agent = DepParseAgent(
        model=model,
        system_prompt=templates.AGENTIC_PARSING_SYSTEM_PROMPT,
        max_new_tokens=10000,
        max_iters_per_call=10,
    )

    # Load documents
    treebanks = {}
    ext_dir = repo_root / "data" / "external"
    print(f"Looking in: {ext_dir}  (exists={ext_dir.exists()})")

    for dir in ext_dir.iterdir():
        print(f"  found entry: {dir}  (is_dir={dir.is_dir()})")
        if not dir.is_dir():
            continue # iterdir() also yields files, not just subdirectories
        train_set_paths = list(dir.rglob("*train*.conllu"))
        print(f"    train_set_paths: {train_set_paths}")
        if not train_set_paths:
            continue

        # dir.name corresponds to ISO-3 language code
        treebanks[dir.name] = CoNLL.conll2doc(train_set_paths[0])
            

    # Register tools
    agent.register_tool(*BagOfWordsRetrievalTool(treebanks).as_agent_tool())
    agent.register_tool(*KNNRetrievalTool(treebanks).as_agent_tool())
    agent.register_tool(*StatisticalMorphologyLookupTool(treebanks).as_agent_tool())
    agent.register_tool(*TreeValidationTool.as_agent_tool())

    # Set up Experiment
    experiment = Experiment(
        agent=agent,
        data_root=data_root,
        new_suffix=PARSED_AGENTICALLY,
        log_suffix="--log_wt",
    )

    # Run
    experiment.run()


if __name__ == "__main__":
    main()
