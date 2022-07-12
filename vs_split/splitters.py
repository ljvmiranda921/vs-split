import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import catalogue
import numpy as np
import spacy
from scipy import stats
from sklearn import feature_extraction, neighbors
from spacy.language import Language
from spacy.tokens import Doc, Span
from wasabi import msg

splitters = catalogue.create("vs_split", "splitters", entry_points=True)


def _get_elements_by_idx(l: Iterable, idx: Iterable) -> Iterable:
    return [l[i] for i in idx]


@splitters.register("wasserstein.v1")
def wasserstein(
    X: Iterable,
    y: Iterable,
    test_size: float = 0.2,
    leaf_size: int = 3,
):
    """
    Perform adversarial splitting using a divergence maximization method
    involving Wasserstein distance.

    This method approximates the test split by performing nearest-neighbor
    search on a random centroid. Based on Søgaard, Ebert et al.'s work on
    'We Need to Talk About Random Splits' (EACL 2021).

    X (Iterable): an array of features.
    y (Iterable): an array of labels.
    test_size (float): the number of neighbors to query.
    leaf_size (int): the leaf size parameter for nearest-neighbor search.
        High values are slower, but less memory-heavy computation.

    RETURNS the training and test sets for the feature and labels respectively.
    """
    nn_tree = neighbors.NearestNeighbors(
        n_neighbors=int(test_size * len(X)),
        algorithm="ball_tree",
        leaf_size=leaf_size,
        metric=stats.wasserstein_distance,
    )
    nn_tree.fit(X)

    sampled_point = np.random.randint(
        np.asarray(X).max().max() + 1, size=(1, np.asarray(X).shape[1])
    )
    nearest_neighbors = nn_tree.kneighbors(sampled_point, return_distance=False)
    test_idxs = nearest_neighbors[0]  # query only a single point

    all_idxs = set(range(len(X)))
    train_idxs = all_idxs - set(test_idxs)

    X_train = _get_elements_by_idx(X, train_idxs)
    y_train = _get_elements_by_idx(y, train_idxs)

    X_test = _get_elements_by_idx(X, test_idxs)
    y_test = _get_elements_by_idx(y, test_idxs)

    msg.text(f"Sizes after split: train ({len(X_train)}), test ({len(X_test)})")
    return X_train, y_train, X_test, y_test


@splitters.register("wasserstein-spacy.v1")
def wasserstein_spacy(
    docs: Iterable[Doc],
    test_size: float = 0.2,
    leaf_size: int = 3,
    use_counts: bool = False,
    min_df: Union[int, float] = 0.10,
    n_jobs: Optional[int] = -1,
) -> Tuple[Iterable[Doc], Iterable[Doc]]:
    """
    Perform adversarial splitting using a divergence maximization method
    involving Wasserstein distance (spaCy-compatible).

    This method approximates the test split by performing nearest-neighbor
    search on a random centroid. Based on Søgaard, Ebert et al.'s work on
    'We Need to Talk About Random Splits' (EACL 2021).

    This splitter takes in an iterable of spaCy Docs and outputs its training
    and test partitions.

    docs (List[Doc]): list of spaCy Doc objects to split.
    test_size (float): the number of neighbors to query.
    leaf_size (int): the leaf size parameter for nearest-neighbor search.
        High values are slower, but less memory-heavy computation.
    use_counts (bool): Use count vectors instead of initialized vectors. If no
        vectors were found, the count vectors are automatically used.
    min_df (Union[int,float]): Remove terms that appear too infrequently given a threshold.
    n_jobs (Optional[int]): Number of parallel jobs to run for neighbor search

    RETURNS the training and test spaCy Doc objects
    """
    if not isinstance(docs[0], Doc):
        # Just check the first element
        msg.fail("Not all elements in `docs` is of type spacy.tokens.Doc", exits=1)

    nn_tree = neighbors.NearestNeighbors(
        n_neighbors=int(test_size * len(docs)),
        algorithm="ball_tree",
        leaf_size=leaf_size,
        metric=stats.wasserstein_distance,
        n_jobs=n_jobs,
    )
    word_vectors = np.asarray([doc.vector for doc in docs])
    if word_vectors.shape[1] == 0:
        msg.warn(
            f"The Doc objects don't contain any vectors (shape: {word_vectors.shape}) "
            "We will use TF-IDF instead."
        )
    if word_vectors.shape[1] == 0 or use_counts:
        texts = [doc.text for doc in docs]
        vectorizer = feature_extraction.text.CountVectorizer(
            dtype=np.int8, min_df=min_df
        )
        text_counts = vectorizer.fit_transform(texts)
        word_vectors = text_counts.todense()

    msg.text(f"Performing nearest neighbor search (shape={word_vectors.shape})")
    nn_tree.fit(word_vectors)

    sampled_point = np.random.randint(
        np.asarray(word_vectors).max().max() + 1,
        size=(1, np.asarray(word_vectors).shape[1]),
    )
    nearest_neighbors = nn_tree.kneighbors(sampled_point, return_distance=False)
    test_idxs = nearest_neighbors[0]  # query only a single point

    all_idxs = set(range(len(docs)))
    train_idxs = all_idxs - set(test_idxs)

    docs_train = _get_elements_by_idx(docs, train_idxs)
    docs_test = _get_elements_by_idx(docs, test_idxs)
    msg.text(f"Sizes after split: train ({len(docs_train)}), test ({len(docs_test)})")
    return docs_train, docs_test


@splitters.register("doc-length.v1")
def doc_length(
    docs: Iterable[Doc],
    test_size: Optional[float] = 0.1,
    length_threshold: Optional[int] = None,
):
    """
    Heuristic split based on document length

    By default, it looks for a sentence length threshold, and puts all the long
    sentences in the test split. The threshold is chosen so that approximately
    10% of the data ends up in the test set.

    You can also override the threshold by passing a value in the
    `length_threshold` parameter.

    docs (List[Doc]): list of spaCy Doc objects to split.
    test_size (Optional[float]): the size of the test set for determining the split.
    length_threshold (Optional[int]): arbitrary length to split the dataset against.

    RETURNS the training and test spaCy Doc objects.
    """
    doc_lengths = [len(doc) for doc in docs]
    if not length_threshold:
        length_threshold = np.percentile(doc_lengths, 100 - (test_size * 100))
        msg.text(f"Splitting the dataset at doc length {int(length_threshold)}")

    all_idxs = set(range(len(docs)))
    # fmt: off
    test_idxs = [
        idx 
        for idx, length in enumerate(doc_lengths) 
        if length >= int(length_threshold)
    ]
    # fmt: on
    train_idxs = all_idxs - set(test_idxs)
    docs_train = _get_elements_by_idx(docs, train_idxs)
    docs_test = _get_elements_by_idx(docs, test_idxs)
    msg.text(f"Sizes after split: train ({len(docs_train)}), test ({len(docs_test)})")

    if len(docs_test) == 0:
        msg.warn("Test set contains no elements!")
    if len(docs_test) >= len(docs_train):
        msg.warn(
            "Test set has a larger size than the train set!"
            f" {len(docs_test)} >= {len(docs_train)}"
        )

    return docs_train, docs_test


@splitters.register("morph-attrs-split.v1")
def morph_attrs_split(
    docs: Iterable[Doc], attrs: List[str] = ["Number", "Person"], test_size: float = 0.2
):
    """
    Perform a heuristic split based on morphological attributes.

    This method is loosely-based on the paper: '(Un)solving Morphological
    Inflection: Lemma Overlap Artificially Inflates Models' Performance' by
    Goldman et. al (ACL 2022). However, instead of focusing solely on lemma
    splits, this method uses morphological attributes. The main motivation is
    because splitting on lemma doesn't translate on standard texts.

    docs (List[Doc]): list of spaCy Doc objects to split.
    attrs (List[str]): morphological attributes to split against.
    test_size (float): the size of the test set for determining the split.

    RETURNS the training and test spaCy Doc objects
    """

    def _get_attr_counts(doc: Doc) -> Counter:
        attrs_freq_per_doc = []
        for token in doc:
            attrs_freq_per_doc.extend(list(token.morph.to_dict().keys()))
        attrs_freq = Counter(attrs_freq_per_doc)
        return attrs_freq

    freqs = []
    for doc in docs:
        ctr = _get_attr_counts(doc)
        # The proceeding line gives 0 if the attr doesn't exist in a particular doc
        # This is due to Counter's behaviour. We don't need to handle cases where key
        # is not in the dictionary
        freq = [ctr[attr] for attr in attrs]
        freqs.append(freq)

    freqs = np.asarray(freqs)
    threshold = np.percentile(freqs, 100 - (test_size * 100), axis=0)

    if not np.any(freqs):
        msg.fail(
            "No MORPH frequencies found. It's probably because you haven't run "
            "a trained pipeline through your texts. Try nlp.pipe([doc.text for "
            "doc in docs]) where nlp is a trained spaCy model (e.g. en_core_web_md).",
            exits=1,
        )

    def _format_msg(attrs, threshold) -> str:
        _msg = [(a, t) for a, t in zip(attrs, threshold)]
        _msg = [f"'{a}' ({int(t)})" for a, t in _msg]
        msg = ", ".join(_msg)
        return msg

    msg.text(f"Splitting on the following threshold/s: {_format_msg(attrs, threshold)}")

    # Get indices based on the computed threshold
    all_idxs = set(range(len(docs)))
    test_idxs = (freqs >= threshold).all(axis=1).nonzero()[0]
    train_idxs = all_idxs - set(test_idxs)

    actual_test_size = round(len(test_idxs) / len(docs), 2)
    msg.text(
        f"Found {len(test_idxs)} documents ({actual_test_size}) "
        "that satisfy the split condition."
    )
    if actual_test_size < test_size:
        msg.warn(
            f"Desired test size ({test_size}) unachieved. Consider "
            "loosening the split conditions for the MORPH attributes."
        )

    docs_train = _get_elements_by_idx(docs, train_idxs)
    docs_test = _get_elements_by_idx(docs, test_idxs)
    if len(docs_test) == 0:
        msg.warn("Test set contains no elements!")
    msg.text(f"Sizes after split: train ({len(docs_train)}), test ({len(docs_test)})")
    return docs_train, docs_test


@splitters.register("entity-switch.v1")
def entity_switch(
    docs: Iterable[Doc],
    patterns: Optional[Dict[str, List[str]]] = None,
    patterns_file: Optional[Path] = None,
    test_size: Optional[float] = None,
    lang: Optional[str] = None,
):
    """
    Manually perturb the test set by switching entities based on a given
    dictionary of patterns.

    This work is based on the paper, 'Entity-Switched Datasets - An Approach to
    Auditing the In-Domain Robustness of Named Entity Recognition Models' by
    Agarwal et al., (ACL 2021). You can control which entity labels are switched
    using a patterns dictionary.

    The patterns dictionary should have the ent label as the key and a list of
    strings as its values. For example, if we want to switch all 'ORG' entities
    in the original document with values such as 'Bene Gesserit', 'Landsraad',
    or 'Spacing Guild', then we should provide a dictionary that look like this:

    patterns = {'ORG': ['Bene Gesserit', 'Landsraad', 'Spacing Guild']}

    You can add as many patterns or entity labels in the dictionary. The pattern chosen
    for substitution is done via random.choice.

    Note that for PERSON names, this splitter does not differentiate between
    first or full names. It just performs a drop-in replacement.

    Implementation-wise, the entity switching is done by recreating the spaCy
    Doc object.  Note that the resulting Docs will only include the text and the
    entity annotations. Any information from the previous pipeline (MORPHS,
    etc.) will be lost.

    docs (List[Doc]): list of spaCy Doc objects to split.
    patterns: Optional[Dict[str, List[str]]]: dictionary of patterns for substitution.
    patterns_file: Optional[Path]: alternative way of providing patterns via a JSON file.
    test_size: Optional[float]: if provided, then the docs will be split further. Since entity-switching
        is only needed for the test set, you can just pass the test documents in this function.
    lang: Optional[str]: the language code to use for recreating the spaCy doc.

    RETURNS the training and test spaCy Doc objects
    """
    if not lang:
        d = random.choice(docs)
        lang = d[0].vocab.lang
        msg.text(f"Naively detected 'lang' as '{lang}'")

    if test_size:
        msg.text(
            # fmt: off
            f"Test size was provided ({test_size}), " 
            "will split the docs accordingly."
            # fmt: on
        )
        num_train = int(len(docs) * (1 - test_size))
        train_docs, _test_docs = docs[:num_train], docs[num_train:]
    else:
        msg.text(
            "Test size not provided, will assume that all docs passed "
            "were test documents. Will return an empty list for training docs "
            "([], List[Docs])."
        )
        train_docs, _test_docs = [], docs

    if patterns_file:
        # Parameter 'patterns_file' will always override 'patterns'
        with patterns_file.open() as f:
            patterns = json.load(patterns)

    if not patterns:
        msg.fail(
            "Patterns file for entity-switching is empty! Did you "
            "provide a JSONL `patterns_file` or passed a dictionary in "
            "the `patterns` parameter?",
            exits=1,
        )

    def _replace_ents(nlp: Language, doc: Doc, patterns: Dict[str, List[str]]) -> Doc:
        nlp.add_pipe("merge_entities")
        merged_ents = [(t.text, t.ent_type_) for t in nlp(doc)]

        new_merged_ents: List[Tuple[str, str]] = []
        for text, ent in merged_ents:
            if ent and (ent in patterns.keys()):
                new_text = random.choice(patterns[ent])
                new_merged_ents.append((new_text, ent))
            else:
                new_merged_ents.append((text, ent))

        new_texts = [text for text, _ in new_merged_ents]
        new_docs: List[Doc] = list(nlp.pipe(new_texts))
        for new_doc, (_, ent) in zip(new_docs, new_merged_ents):
            if ent:
                new_doc.set_ents([Span(new_doc, 0, len(new_doc), ent)])

        merged_doc = Doc.from_docs(new_docs)
        return merged_doc

    test_docs: List[Doc] = []
    for test_doc in _test_docs:
        nlp = spacy.blank(lang)
        sub_test_doc = _replace_ents(nlp, doc=test_doc, patterns=patterns)
        test_docs.append(sub_test_doc)

    return train_docs, test_docs
