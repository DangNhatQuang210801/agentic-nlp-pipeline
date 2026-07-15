"""Smoke-test the existing direct dependency-parsing harness on Marathi.

Drop this into `scripts/` (per the project convention: scripts/ is for
stand-alone code that isn't called from elsewhere).

Usage:
    python scripts/run_marathi_direct_parsing.py --split test
    python scripts/run_marathi_direct_parsing.py --split test --limit 5 --verbose

Prerequisites:
    1. Fetch the treebank (data/ is gitignored by design, same as Vietnamese):

       git clone --depth 1 https://github.com/UniversalDependencies/UD_Marathi-UFAL.git \
         data/marathi/raw/UD_Marathi-UFAL

    2. `direct_dep_parsing` currently can't be imported as a package module:
       agentic_nlp_pipeline/prompting/harnesses.py uses
           from templates import DIRECT_PARSING_SYSTEM_PROMPT
           from parsing import parse_tree_from_text, insert_tree_into_sentence
       which only resolve when harnesses.py is run directly as __main__.
       Fix (one line each) before this script will import cleanly:
           from .templates import DIRECT_PARSING_SYSTEM_PROMPT
           from .parsing import parse_tree_from_text, insert_tree_into_sentence

    3. agentic_nlp_pipeline/prompting/parsing.py has an off-by-one bug in
       insert_tree_into_sentence:
           sent.words[s_node - 1].head = e_node - 1   # wrong
       should be:
           sent.words[s_node - 1].head = e_node       # head IDs aren't 0-indexed
       Without this fix every predicted HEAD (including root) is shifted by
       one and UAS numbers will be silently wrong. Flag/fix this with
       whoever owns parsing.py before trusting any results from this script.

Note on scoring:
    Task 0.4 (the real UAS/LAS/CLAS evaluation harness) isn't in the repo
    yet, so the `uas()` helper below is a local, throwaway sanity-check
    metric only -- not a substitute for that module. Swap it out once 0.4
    lands instead of extending this one.
"""

import argparse
import csv
from pathlib import Path

from stanza.utils.conll import CoNLL
from stanza.models.common.doc import Sentence

from agentic_nlp_pipeline import LocalModel, validate_sentence
from agentic_nlp_pipeline.prompting.harnesses import direct_dep_parsing

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "externaml" / "mar"
RESULTS_DIR = REPO_ROOT / "results" / "marathi"


def strip_gold(sent: Sentence) -> Sentence:
    """Build a query Sentence with only ID/FORM, no gold HEAD/DEPREL.

    Mirrors the plain {id, text} construction already used in
    harnesses.py's own __main__ example. Constructing a Sentence this way
    leaves `.head` as None for every word, so if the model fails to cover
    some token, that token is scored wrong -- it can't silently inherit
    the gold answer the way it would if we reused `sent` directly.
    """
    return Sentence([{"id": w.id, "text": w.text} for w in sent.words])


def uas(pred: Sentence, gold: Sentence) -> tuple[int, int]:
    """Local, throwaway unlabeled-attachment count.

    Not the project's evaluation harness (task 0.4) -- just enough to
    sanity-check that the pipeline is doing something sensible on Marathi.
    """
    correct = sum(1 for p, g in zip(pred.words, gold.words) if p.head == g.head)
    return correct, len(gold.words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--max-new-tokens", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None, help="Only run the first N sentences (debugging)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    conllu_path = DATA_DIR / f"mr_ufal-ud-{args.split}.conllu"
    if not conllu_path.exists():
        raise FileNotFoundError(
            f"{conllu_path} not found. Fetch the treebank first:\n"
            f"  git clone --depth 1 https://github.com/UniversalDependencies/UD_Marathi-UFAL.git "
            f"{DATA_DIR}"
        )

    doc = CoNLL.conll2doc(input_file=str(conllu_path))
    sentences = doc.sentences[: args.limit] if args.limit else doc.sentences
    print(f"Loaded {len(sentences)} Marathi sentences from {conllu_path.name}")

    model = LocalModel(args.model)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"direct_parsing_{args.split}.csv"

    total_correct = total_tokens = 0
    n_valid = 0

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sent_id", "n_tokens", "correct_heads", "valid_tree", "validity_msg"])

        for gold in sentences:
            query = strip_gold(gold)

            pred = direct_dep_parsing(
                model=model,
                sent=query,
                max_new_tokens=args.max_new_tokens,
                verbose=args.verbose,
            )

            try:
                valid, msg = validate_sentence(pred)
            except TypeError:
                # Some HEAD values were never predicted and are still None --
                # validate_sentence assumes ints, so this needs to be caught
                # here rather than crashing the whole run.
                valid, msg = False, "Incomplete tree: some HEAD values were never predicted."

            correct, n_tokens = uas(pred, gold)
            total_correct += correct
            total_tokens += n_tokens
            n_valid += int(valid)

            sent_id = getattr(gold, "sent_id", "")
            writer.writerow([sent_id, n_tokens, correct, valid, msg])
            print(f"{sent_id!s:>6}  UAS {correct}/{n_tokens}  valid={valid}")

    print("\n" + "=" * 50)
    print(f"Sentences run     : {len(sentences)}")
    print(f"Valid trees       : {n_valid}/{len(sentences)}")
    if total_tokens:
        print(f"Local UAS (proxy) : {total_correct}/{total_tokens} ({total_correct / total_tokens:.1%})")
    print(f"Results written to: {out_path}")
    print("NB: this UAS is a local smoke-test metric, not the official eval harness (task 0.4).")


if __name__ == "__main__":
    main()