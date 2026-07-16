from pathlib import Path

from agentic_nlp_pipeline import DepParseAgent, LocalModel, Experiment
from agentic_nlp_pipeline.experiment import PARSED_DIRECTLY
from agentic_nlp_pipeline.agentic import templates


def main():
    # Find root dir
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data" / "processed"

    # Set up experiment
    model = LocalModel("unsloth/Qwen3.5-9B-GGUF", gguf_file="Qwen3.5-9B-Q4_K_M.gguf", enable_thinking=True)
    agent = DepParseAgent(model, templates.DIRECT_PARSING_SYSTEM_PROMPT, 10000, 10)
    experiment = Experiment(agent, data_root, PARSED_DIRECTLY)

    # Run
    experiment.run()


if __name__ == "__main__":
    main()
