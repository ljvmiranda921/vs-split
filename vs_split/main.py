from typing import Iterable

from spacy.tokens import Doc
from wasabi import msg

from vs_split.splitters import splitters as splitter_catalogue


def train_test_split(X: Iterable, y: Iterable, split_id: str, **attrs):
    splitter = splitter_catalogue.get(split_id)
    return splitter(X, y, **attrs)


def spacy_train_test_split(docs: Iterable[Doc], split_id: str, **attrs):
    splitter = splitter_catalogue.get(split_id, **attrs)
