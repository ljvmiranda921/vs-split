"""Demo for splitting by divergence maximization"""

import itertools
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import spacy
import typer
from spacy.cli._util import parse_config_overrides
from spacy.cli.evaluate import evaluate as spacy_evaluate
from spacy.cli.train import train as spacy_train
from spacy.tokens import Doc, DocBin
from vs_split import spacy_train_test_split
from wasabi import msg

DEFAULT_SPLITS = ["wasserstein-spacy.v1"]
CORPUS_PATH = Path().parent / "corpus"


def main(
    config_path: Path,
    display_size: int = 3,
    splitters: List[str] = DEFAULT_SPLITS,
    fit_model: bool = True,
    vectors: str = "en_core_web_lg",
    use_gpu: int = 0,
    max_steps: int = 20000,
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
            adv_scores = _fit_and_evaluate_model(
                ntrain,
                ntest,
                config_path,
                vectors=vectors,
                use_gpu=use_gpu,
                max_steps=max_steps,
            )
            std_scores = _fit_and_evaluate_model(
                traindev,
                test,
                config_path,
                vectors=vectors,
                use_gpu=use_gpu,
                max_steps=max_steps,
            )
            print(_format_table(adv_scores, std_scores, split_id))


def _fit_and_evaluate_model(
    train: List[Doc],
    test: List[Doc],
    config_path: Path,
    use_gpu: int = 0,
    vectors: str = "en_core_web_lg",
    max_steps: int = 20000,
) -> Dict[str, Any]:
    """Fit a NER model and evaluate it

    Instead of working with the registered architectures, I decided to just mimic
    what happens in spaCy CLI while in a temporary directory.
    """

    def _split_train_dev(
        traindev: List[Doc], split_size: float = 0.8
    ) -> Tuple[List[Doc], List[Doc]]:
        train_size = int(len(traindev) * split_size)
        return traindev[:train_size], traindev[train_size:]

    def _save_docs_to_tmp(
        train: List[Doc], dev: List[Doc], test: List[Doc], output_dir: Path
    ) -> List[str]:
        datasets = {
            "train.spacy": train,
            "dev.spacy": dev,
            "test.spacy": test,
        }
        for filename, docs in datasets.items():
            doc_bin = DocBin(docs=docs)
            doc_bin.to_disk(output_dir / filename)

        filepaths = [output_dir / filename for filename in datasets.keys()]
        return filepaths

    ntrain, ndev = _split_train_dev(train)
    with tempfile.TemporaryDirectory() as tmp_dir:
        msg.text(f"Performing training and evaluation in {tmp_dir}...")
        # Setup the dataset
        tmp_dir_path = Path(tmp_dir)
        tmp_filepaths = _save_docs_to_tmp(ntrain, ndev, test, tmp_dir_path)
        train_fp, dev_fp, test_fp = tmp_filepaths
        model_path = tmp_dir_path / "output"
        # Train model
        msg.text(f"Training model (will be saved at {str(model_path)})")
        spacy_train(
            config_path=config_path,
            output_path=model_path,
            overrides={
                "paths.train": str(train_fp),
                "paths.dev": str(dev_fp),
                "paths.vectors": vectors,
                "training.max_steps": max_steps,
            },
            use_gpu=use_gpu,
        )
        # Evaluate model
        scores = spacy_evaluate(
            model=model_path / "model-best",
            data_path=test_fp,
            use_gpu=use_gpu,
        )

    return scores


def _format_table(
    adv_scores: Dict[str, Any], std_scores: Dict[str, Any], split_id: str
) -> str:
    header = ["Split", "ENTS_P", "ENTS_R", "ENTS_F"]
    adv = [adv_scores["ents_p"], adv_scores["ents_r"], adv_scores["ents_f"]]
    adv = [round(s, 2) for s in adv]
    std = [std_scores["ents_p"], std_scores["ents_r"], std_scores["ents_f"]]
    std = [round(s, 2) for s in std]
    data = [[split_id] + adv, ["Standard"] + std]
    table = msg.table(data, header=header, divider=True)
    return table


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
    typer.run(main)
