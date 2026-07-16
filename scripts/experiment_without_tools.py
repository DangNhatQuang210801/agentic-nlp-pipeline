from pathlib import Path

from agentic_nlp_pipeline.agentic import DepParseAgent, LlamaCppModel, Experiment
from agentic_nlp_pipeline.experiment import PARSED_DIRECTLY
from agentic_nlp_pipeline.agentic import templates


def main():
    # Find root dir
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data" / "processed"

    # Set up experiment
    model = LlamaCppModel()
    agent = DepParseAgent(model, templates.DIRECT_PARSING_SYSTEM_PROMPT, 10000, 10)
    experiment = Experiment(agent, data_root, PARSED_DIRECTLY)

    # Run
    experiment.run()


if __name__ == "__main__":
    main()
