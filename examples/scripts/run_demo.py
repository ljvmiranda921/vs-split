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
METRICS = ["ents_p", "ents_r", "ents_f"]

DEFAULT_TRAIN = CORPUS_PATH / "en-wikineural-train.spacy"
DEFAULT_DEV = CORPUS_PATH / "en-wikineural-dev.spacy"
DEFAULT_TEST = CORPUS_PATH / "en-wikineural-test.spacy"


def main(
    config_path: Path,
    train_dataset: Path = DEFAULT_TRAIN,
    dev_dataset: Path = DEFAULT_DEV,
    test_dataset: Path = DEFAULT_TEST,
    splitters: List[str] = DEFAULT_SPLITS,
    fit_model: bool = True,
    vectors: str = "en_core_web_lg",
    use_gpu: int = 0,
    max_steps: int = 20000,
):
    msg.info(f"Splitters: {','.join(splitters)}")
    train = _get_docs(train_dataset)
    dev = _get_docs(dev_dataset)
    test = _get_docs(test_dataset)
    rows = []  # keep track of the scores for reporting

    traindev = _combine_docs(train, dev)
    if fit_model:
        msg.info("Getting baseline performance for standard split")
        std_scores = _fit_and_evaluate_model(
            traindev,
            test,
            config_path,
            vectors=vectors,
            use_gpu=use_gpu,
            max_steps=max_steps,
        )
        rows.append(["standard"] + [std_scores[metric] for metric in METRICS])

    for split_id in splitters:

        msg.divider(text=split_id)
        dataset = _combine_docs(train, dev, test)
        ntrain, ntest = spacy_train_test_split(
            dataset, split_id=split_id, n_jobs=-1, min_df=0.10
        )

        if fit_model:
            adv_scores = _fit_and_evaluate_model(
                ntrain,
                ntest,
                config_path,
                vectors=vectors,
                use_gpu=use_gpu,
                max_steps=max_steps,
            )
            rows.append([split_id] + [adv_scores[metric] for metric in METRICS])

    # Report scores
    if rows:
        header = ["Split", "ENTS_P", "ENTS_R", "ENTS_F"]
        rows = _round_up_scores(rows)
        msg.table(rows, header=header, divider=True)


def _fit_and_evaluate_model(
    train: List[Doc],
    test: List[Doc],
    config_path: Path,
    use_gpu: int = 0,
    vectors: str = "en_core_web_lg",
    max_steps: int = 20000,
) -> Dict[str, Any]:
    """Fit a NER model and evaluate it

    Instead of working with the registered architectures, I decided to just
    mimic what happens in spaCy CLI while in a temporary directory. I also
    exposed a few parameters that I find important for quick-testing

    train (List[Doc]): the spaCy Doc objects that will be used for training.
    test (List[Doc]): the spaCy Doc objects that will be used for testing.
    config_path (Path): path to the training configuration file.
    use_gpu (int): GPU ID for training. Use -1 for CPU.
    vectors (str): Name of base model to initialize the training with.
    max_steps (int): Maximum number of steps for training.

    RETURNS (Dict[str, Any]) a dictionary of scores
    """

    def _split_train_dev(
        traindev: List[Doc], split_size: float = 0.8
    ) -> Tuple[List[Doc], List[Doc]]:
        """Split the training set into train and dev"""
        train_size = int(len(traindev) * split_size)
        return traindev[:train_size], traindev[train_size:]

    def _save_docs_to_tmp(
        train: List[Doc], dev: List[Doc], test: List[Doc], output_dir: Path
    ) -> List[str]:
        """Save Doc objects as a .spacy file in an output directory"""
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


def _round_up_scores(rows: List[List[Any]], ndigits: int = 2) -> List[List[Any]]:
    """Reduce scores into 2-decimal numbers for readability"""
    new_table = []
    for row in rows:
        new_row = []
        for el in row:
            new_row.append(round(el, ndigits) if isinstance(el, float) else el)
        new_table.append(new_row)
    return new_table


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


if __name__ == "__main__":
    typer.run(main)
