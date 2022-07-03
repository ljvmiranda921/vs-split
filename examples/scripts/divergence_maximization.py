"""Demo for splitting by divergence maximization"""

import itertools
import random
from pathlib import Path
from typing import List

import spacy
from spacy.tokens import Doc, DocBin
from spacy.cli.train import train as spacy_train
from wasabi import msg

from vs_split import spacy_train_test_split

DEFAULT_SPLITS = ["wasserstein-spacy.v1"]
CORPUS_PATH = Path().parent / "corpus"


def main(
    config_path: Path,
    display_size: int = 3,
    splitters: List[str] = DEFAULT_SPLITS,
    fit_model: bool = True,
):
    msg.info(f"Splitters: {','.join(splitters)}")

    msg.divider(text="Divergence maximization")
    msg.info(
        "This script runs a demo of the divergence maximization split. "
        "On NLP datasets such as Wikineural (en) and ConLL2003. You can configure "
        "the splits to use by setting the --splitters parameter."
    )

    for split_id in splitters:

        # Wikineural dataset (english)
        msg.divider(text="Dataset: en-wikineural", char="-")
        # TODO: specify paths
        train = _get_docs(CORPUS_PATH / "en-wikineural-train.spacy")
        dev = _get_docs(CORPUS_PATH / "en-wikineural-dev.spacy")
        test = _get_docs(CORPUS_PATH / "en-wikineural-test.spacy")

        dataset = _combine_docs(train, dev, test)
        traindev = _combine_docs(train, dev)
        ntrain, ntest = spacy_train_test_split(
            dataset, split_id=split_id, n_jobs=-1, min_df=0.10
        )
        _display_train_test(traindev, test, ntrain, ntest, display_size)

        if fit_model:
            # TODO
            spacy_train(config_path=config_path, output_path=output_path, overrides={})


def _get_docs(docbin_path: Path) -> List[Doc]:
    """Read Doc objects given a Docbin path"""
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
    """Report the split train test partitions"""

    def _format_docs(docs: List[Doc], title: str):
        random.shuffle(docs)
        texts = [doc.text for doc in docs[:display_size]]
        for idx, text in enumerate(texts):
            msg.divider(f"{title} ({idx})", char=".")
            msg.text(f"{text}\n")

    msg.info("Sample texts from the previous split")
    _format_docs(old_train, "Old train example")
    _format_docs(old_test, "Old test example")

    msg.info("Sample texts from the new split")
    _format_docs(new_train, "New train example")
    _format_docs(new_test, "New test example")


if __name__ == "__main__":
    main()
