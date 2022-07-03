from typing import Iterable, Optional, Tuple, Union

import catalogue
import numpy as np
from scipy import stats
from sklearn import feature_extraction, neighbors
from spacy.tokens import Doc
from wasabi import msg

splitters = catalogue.create("vs_split", "splitters", entry_points=True)


def _get_elements_by_idx(l: Iterable, idx: Iterable) -> Iterable:
    return [l[i] for i in idx]


@splitters.register("wasserstein.v1")
def wasserstein(
    X: Iterable,
    y: Iterable,
    test_size: float = 0.2,
    n_trials: int = 1,
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
    n_trials (int): number of test sets requested.
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

    test_idxs = []
    for trial in range(n_trials):
        msg.text(f"Trial set: {trial}")
        sampled_point = np.random.randint(
            np.asarray(X).max().max() + 1, size=(1, np.asarray(X).shape[1])
        )
        nearest_neighbors = nn_tree.kneighbors(sampled_point, return_distance=False)
        nearest_neighbor = nearest_neighbors[0]  # query only a single point
        test_idxs.append(nearest_neighbor)

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
    n_trials: int = 1,
    leaf_size: int = 3,
    use_counts: bool = False,
    min_df: Union[int, float] = 1,
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
    n_trials (int): number of test sets requested.
    leaf_size (int): the leaf size parameter for nearest-neighbor search.
        High values are slower, but less memory-heavy computation.
    use_counts (bool): Use count vectors instead of spaCy docs.
    min_df (Union[int,float]): Remove terms that appear too infrequently given a threshold.

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
    )
    word_vectors = np.asarray([doc.vector for doc in docs])
    if word_vectors.shape[1] == 0:
        msg.warn(
            f"The Doc objects don't contain any vectors (shape: {word_vectors.shape}) "
            "We will use TF-IDF instead."
        )
    if word_vectors.shape[1] == 0 or use_counts:
        texts = [doc.text for doc in docs]
        vectorizer = feature_extraction.text.CountVectorizer(dtype=np.int8)
        text_counts = vectorizer.fit_transform(texts)
        word_vectors = text_counts.todense()

    nn_tree.fit(word_vectors)

    test_idxs = []
    for trial in range(n_trials):
        msg.text(f"Trial set: {trial}")
        sampled_point = np.random.randint(
            np.asarray(word_vectors).max().max() + 1,
            size=(1, np.asarray(word_vectors).shape[1]),
        )
        nearest_neighbors = nn_tree.kneighbors(sampled_point, return_distance=False)
        nearest_neighbor = nearest_neighbors[0]  # query only a single point
        test_idxs.append(nearest_neighbor)

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
