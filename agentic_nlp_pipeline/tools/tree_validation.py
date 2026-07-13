from typing import Callable
from stanza.models.common.doc import Sentence

from .utils import token_dicts_to_sentence


class TreeValidationTool:
    schema = {
        "type": "function",
        "function": {
            "name": "validate_dependency_tree",
            "description": (
                "Check whether the sentence's dependency tree adhears to a number of formal constraints. This is something to be checked befor submitting a final answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sent": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "head": {"type": "integer"},
                            },
                            "required": ["id", "head"],
                        },
                        "description": "The dependency tree as an array of token objects.",
                    },
                },
                "required": ["sent"],
            },
        },
    }

    @staticmethod
    def _has_valid_heads(sent: Sentence) -> tuple[bool, str]:
        """Check if the HEAD attributes of a Sentence are in the right range.

        Args:
            sent: Some Stanza Sentence.

        Returns:
            has_valid_heads: Whether or not the HEADs are all valid.
            msg: A detailed success or error message.
        """
        invalid_ids = []
        invalid_heads = []
        max_id = max(w.id for w in sent.words)

        for word in sent.words:
            if word.head > max_id:
                invalid_ids.append(word.id)
                invalid_heads.append(word.head)

        n_invalid = len(invalid_ids)

        if n_invalid == 0:
            msg = "All HEAD values are in the right range."
        elif n_invalid == 1:
            # GenAI has been used to polish this error message.
            msg = (
                f"Invalid HEAD value at word index {invalid_ids[0]}: "
                f"expected 0 (root) or a value between 1 and {max_id} "
                f"(the number of words in the sentence), got {invalid_heads[0]}."
            )
        else:
            msg = (
                f"Invalid HEAD values at word indices {invalid_ids}: "
                f"expected 0 (root) or a value between 1 and {max_id} "
                f"(the number of words in the sentence), got {invalid_heads}."
            )

        has_valid_heads = n_invalid == 0
        return has_valid_heads, msg

    @staticmethod
    def _has_single_root(sent: Sentence) -> tuple[bool, str]:
        """Check if a Sentence as one and only one root.

        Args:
            sent: Some Stanza Sentence.

        Returns:
            has_single_root: Whether or not the HEADs are all valid.
            msg: A detailed success or error message.
        """
        n_roots = sum(1 if w.head == 0 else 0 for w in sent.words)
        roots = [w.id for w in sent.words if w.head == 0]

        if n_roots == 1:
            msg = "The dependency tree has exactly one root (as it should)."
        elif n_roots == 0:
            msg = "The dependency tree does not have a root, but is should have."
        else:
            msg = (
                f"The tree has {n_roots} roots (indices {roots}), "
                "but it should only have one."
            )

        has_single_root = n_roots == 1
        return has_single_root, msg

    @staticmethod
    def _is_acyclic(sent: Sentence) -> tuple[bool, str]:
        """Check if the dependency graph of a Sentence contains any cycles.

        Args:
            sent: Some Stanza Sentence.

        Returns:
            is_acyclic: Whether or not the dependency graph is acyclic.
            msg: A detailed success or error message.
        """
        for word in sent.words:
            head = word.id
            chain = [head]

            while (head := sent.words[head - 1].head) not in chain + [0]:
                chain.append(head)

            # Return early if a cycle has been found
            final_head = sent.words[chain[-1] - 1].head
            if final_head != 0:
                cycle_len = len(chain)
                if cycle_len == 1:
                    msg = f"Found an isolated word at index {chain[-1]}."
                else:
                    msg = (
                        f"Found a cycle in the dependency graph even though it should "
                        f"be acyclic. The cycle is made up of the word indices {chain}."
                    )
                is_acyclic = False
                return is_acyclic, msg

        is_acyclic = True
        msg = "The dependency tree is acyclic (as it should be)."
        return is_acyclic, msg

    @classmethod
    def validate_dependency_tree(cls, sent: list[dict[str, str | int]]) -> str:
        """Check if a Sentence has a valid dependency tree.

        Args:
            sent: Some Stanza Sentence.

        Returns:
            valid: Whether or not the dependency tree is valid.
            msg: A detailed success or error message.
        """
        for check in [cls._has_valid_heads, cls._has_single_root, cls._is_acyclic]:
            valid, msg = check(token_dicts_to_sentence(sent))
            if not valid:
                return msg

        valid = True
        msg = "The sentences dependency graph has a valid tree structure."
        return msg

    @classmethod
    def as_agent_tool(cls) -> tuple[dict, Callable]:
        return cls.schema, cls.validate_dependency_tree
