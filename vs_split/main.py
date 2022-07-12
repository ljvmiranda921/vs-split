from typing import Iterable

from spacy.tokens import Doc
from wasabi import msg

from vs_split.splitters import splitters as splitter_catalogue


def train_test_split(X: Iterable, y: Iterable, split_id: str, **attrs):
    """Split a dataset into its training and testing partitions

    X (Iterable): an iterable of features, preferably a numpy.ndarray.
    y (Iterable): an iterable of labels, preferably a numpy.ndarray.
    split_id (str): the type of split to use.
    """
    splitter = splitter_catalogue.get(split_id)
    return splitter(X, y, **attrs)


def spacy_train_test_split(docs: Iterable[Doc], split_id: str, **attrs):
    """Split a list of spaCy Doc objects into its training and testing partitions

    docs (Iterable[Doc]): list of spaCy Doc objects to split.
    split_id (str): the type of split to use.
    """
    splitter = splitter_catalogue.get(split_id)
    return splitter(docs, **attrs)
