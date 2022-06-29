from typing import Iterable, Tuple

import catalogue
import numpy as np
from scipy import stats
from sklearn import neighbors
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
            np.array(X).max().max() + 1, size=(1, np.array(X).shape[1])
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
def wasserstein(
    docs: Iterable[Doc],
    test_size: float = 0.2,
    n_trials: int = 1,
    leaf_size: int = 3,
) -> Tuple[Iterable[Doc], Iterable[Doc]]:
    """
    Perform adversarial splitting using a divergence maximization method
    involving Wasserstein distance (spaCy-compatible).

    This method approximates the test split by performing nearest-neighbor
    search on a random centroid. Based on Søgaard, Ebert et al.'s work on
    'We Need to Talk About Random Splits' (EACL 2021).

    This splitter takes in an iterable of spaCy Docs and outputs its training
    and test partitions.

    test_size (float): the number of neighbors to query.
    n_trials (int): number of test sets requested.
    leaf_size (int): the leaf size parameter for nearest-neighbor search.
        High values are slower, but less memory-heavy computation.

    RETURNS the training and test spaCy Docs
    """
    nn_tree = neighbors.NearestNeighbors(
        n_neighbors=int(test_size * len(docs)),
        algorithm="ball_tree",
        leaf_size=leaf_size,
        metric=stats.wasserstein_distance,
    )
    word_vectors = [doc.vector for doc in docs]
    nn_tree.fit(word_vectors)

    test_idxs = []
    for trial in range(n_trials):
        msg.text(f"Trial set: {trial}")
        sampled_point = np.random.randint(
            np.array(word_vectors).max().max() + 1,
            size=(1, np.array(word_vectors).shape[1]),
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
