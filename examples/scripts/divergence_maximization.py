"""Demo for splitting by divergence maximization"""

import itertools
import random
from pathlib import Path
from typing import List

import spacy
import typer
from spacy.tokens import Doc, DocBin
from wasabi import msg

from vs_split import spacy_train_test_split

DEFAULT_SPLITS = ["wasserstein-spacy.v1"]
CORPUS_PATH = Path().parent / "corpus"


def main(display_size: int = 5, splitters: List[str] = DEFAULT_SPLITS):
    msg.divider(text="Divergence maximization")
    msg.info(
        "This script runs a demo of the divergence maximization split. "
        "On NLP datasets such as Wikineural (en) and ConLL2003. You can configure "
        "the splits to use by setting the --splitters parameter."
    )

    for split_id in splitters:

        # Wikineural dataset (english)
        msg.divider(text="en-wikineural", char="-")
        # TODO: specify paths
        train = _get_docs(CORPUS_PATH / "en-wikineural-train.spacy")
        dev = _get_docs(CORPUS_PATH / "en-wikineural-dev.spacy")
        test = _get_docs(CORPUS_PATH / "en-wikineural-test.spacy")

        dataset = _combine_docs(train, dev, test)
        traindev = _combine_docs(train, dev)
        ntrain, ntest = spacy_train_test_split(dataset, split_id=split_id)
        _display_train_test(traindev, test, ntrain, ntest, display_size)


def _get_docs(docbin_path: Path) -> List[Doc]:
    docbin = DocBin().from_disk(docbin_path)
    nlp = spacy.blank("en")
    docs = list(docbin.get_docs(nlp.vocab))
    return docs


def _combine_docs(*args: List[Doc]) -> List[Doc]:
    """Combine list of Doc objects into a single Doc

    Useful to combine training, dev, and test sets into a single dataset
    before adversarial splitting.
    """
    return list(itertools.chain.from_iterable(args))


def _display_train_test(
    old_train: List[Doc],
    old_test: List[Doc],
    new_train: List[Doc],
    new_test: List[Doc],
    display_size: int,
):
    def _show(docs: List[Doc]):
        for doc in random.shuffle(docs)[:display_size]:
            msg.text(doc.text)

    msg.info("Sample texts from the previous split")
    _show(old_train)
    _show(old_test)

    msg.info("Sample texts from the new split")
    _show(new_train)
    _show(new_test)


if __name__ == "__main__":
    main()
