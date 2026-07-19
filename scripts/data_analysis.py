from pathlib import Path

import pandas as pd

from agentic_nlp_pipeline.evaluation import compile_df
from agentic_nlp_pipeline.experiment import (
    LOG_AGENTICALLY,
    LOG_DIRECTLY,
    PARSED_AGENTICALLY,
    PARSED_DIRECTLY,
    UNPARSED,
)


def main():
    repo_root = Path(__file__).resolve().parents[1]

    df_parsed_directly = compile_df(
        repo_root=repo_root,
        gold_suffix=UNPARSED,
        pred_suffix=PARSED_DIRECTLY,
        log_suffix=LOG_DIRECTLY,
        with_tools=False,
    )
    df_parsed_agentically = compile_df(
        repo_root=repo_root,
        gold_suffix=UNPARSED,
        pred_suffix=PARSED_AGENTICALLY,
        log_suffix=LOG_AGENTICALLY,
        with_tools=True,
    )
    df = pd.concat([df_parsed_directly, df_parsed_agentically])

    df.to_csv(repo_root / "data" / "processed" / "parse.csv", sep="\t")
    print(df)


if __name__ == "__main__":
    main()
