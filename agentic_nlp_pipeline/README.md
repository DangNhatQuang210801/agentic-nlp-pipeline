Overview over the directories:
- `evaluation/`: Should contain the methods for measuring annotation accuracy (Task 0.4) and for projectivity checks (Task 0.2).
- `models/`: This contains wrappers around local and hosted models as well as the base class `LanguageModel`.
- `prompting/`: Contains everything needed for prompting: Prompt templates (`templates.py`), methods for parsing the model responses (`parsing.py`) and methods for model orchestration (`harnesses.py`) (Task 1.4).
- `validation/`: Methods for validating trees (should be usable as agent tools later) (Task 0.5).

The public API should get exposed through `__init__.py`.
