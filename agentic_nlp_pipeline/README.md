Overview over the directories:
- `data/`: Right now this contains code for tree validation and projectivity checks. Perhaps it move `tree_validation.py` into a `validation` directory and `projectivity.py` into a `evaluation` directory.
- `models/`: This contains wrappers around local and hosted models as well as the base class `LanguageModel`.
- `prompting/`: Contains everything needed for prompting: Prompt templates (`templates.py`), methods for parsing the model responses (`parsing.py`) and methods for model orchestration (`harnesses.py`).

Not yet created:
- `evaluation/`: Should contain methods for measuring annotation accuracy (Task 0.4) and maybe also the projectivity checks (Task 0.2).
- `validation/`: Methods for validating trees (should be usable as agent tools later) (Task 0.5).

The public API should get exposed through `__init__.py`.
