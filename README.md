# ⚔️ vs-split: a library for creating adversarial splits

Have you ever encountered a problem where **your model works well in your test set
but doesn't perform well in the wild?**  It's likely because your test set does
not reflect the reality of your domain, overestimating your model's performance.[^1]

This library provides **alternative ways to split and sanity-check your datasets**
and ensure they're robust once you deploy them into production.

[^1]: Check out my blog post, [*Your train-test split may be doing you a disservice*](https://ljvmiranda921.github.io/2022/08/30/adversarial-splits/), for a technical overview of this problem.

## ⏳ Installation

You can install `vs-split` via `pip`

```sh
pip install vs-split
```

Or alternatively, you can install from source:

```sh
git clone https://github.com/ljvmiranda921/vs-split
cd vs-split
python setup.py install
```

## 👩‍💻 Usage

The library exposes two main functions: 

- **`train_test_split(X: Iterable, y: Iterable, split_id: str, **attrs)`** that accepts [NumPy arrays](https://numpy.org/doc/stable/reference/generated/numpy.array.html) of your features and labels. You can pass any arbitrary NumPy array or list for splitting.
- **`spacy_train_test_split(docs: Iterable[Doc], split_id: str, **attrs)`** that accepts an iterable of [spaCy Doc objects](https://spacy.io/api/doc).[^2] [spaCy](https://spacy.io) is a Python library for natural language processing and the Doc object is one of its core data structures. This function is useful if you're working on linguistic data.  

For both functions, you can provide the type of split in the `split_id`
parameter (c.f. [splitters catalogue](#splitters-catalogue)) and pass custom
keyword-arguments.

```python
from vs_split import train_test_split, spacy_train_test_split

# For most datasets
X_train, y_train, X_test, y_test = train_test_split(X_data, y_data, split_id="wasserstein.v1")
# For spaCy Doc objects
docs_train, docs_test = spacy_train_test_split(docs, split_id="wasserstein-spacy.v1")
```

> **Note**
> It might look like `vs-split` has a similar API with [scikit-learn's
> `train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html),
> but that's not the case.  Unlike the latter, `vs_split.train_test_split` doesn't expect
> an arbitrary number of iterables, and the keyword parameters are also different.

[^2]: vs-split has first-class support for spaCy. The main reason is that I've been using this for some internal robustness experiments to test some of our [pipeline components](https://spacy.io/usage/processing-pipelines).

### Registering your own splitters

You can also register custom splitters via the `splitters` catalogue. Here's an
example of a splitter, `random-spacy.v1` that splits a list of spaCy Doc objects
given a training set size:

```python
import random
from typing import Iterable

from spacy.tokens import Doc
from vs_split.splitters import splitters

@splitters.register("random-spacy.v1")
def random_spacy(docs: Iterable[Doc], train_size: float):
    random.shuffle(docs)
    num_train = int(len(docs) * train_size)
    train_docs = docs[:num_train]
    test_docs = docs[num_train:]
    return train_docs, test_docs
```

Under the hood, `vs-split` uses
[`catalogue`](https://github.com/explosion/catalogue) to manage the functions
you registered. You are given freedom to return any value / object in your
splitter implementation&mdash;i.e, there's no function that enforces you to
follow the blueprint. However, for consistency, it's advisable to follow the
type signature of the other splitters.

### More examples

You can find more in the
[`examples/`](https://github.com/ljvmiranda921/vs-split/tree/main/examples)
directory. It contains a sample project that runs the [English WikiNeural
dataset](https://paperswithcode.com/dataset/wikineural) on various spaCy
splitters.

## 🎛 API

### <kbd>function</kbd> `train_test_split`

Split a dataset into its training and testing partitions. By default, it should
return the training and testing features and labels respectively. 

| Argument    | Type       | Description                                            |
|-------------|------------|--------------------------------------------------------|
| `*X`        | Iterable   | An iterable of features, preferably a `numpy.ndarray`. |
| `*y`        | Iterable   | An iterable of labels, preferably a `numpy.ndarray`.   |
| `*split_id` | str        | The type of split to use.                              |
| **RETURNS** | Tuple[Iterable[Any], Iterable[Any], Iterable[Any], Iterable[Any]] | The training and testing features and labels (i.e. `X_train`, `y_train`, `X_test`, `y_test`). |


### <kbd>function</kbd> `spacy_train_test_split`

Split a list of spaCy `Doc` objects into its training and testing partitions. By default, it should return the training and test spaCy Doc objects respectively.

| Argument    | Type         | Description                                            |
|-------------|--------------|--------------------------------------------------------|
| `*docs`     | Iterable[Doc]| An iterable of spaCy Doc objects to split.             |
| `*split_id` | str          | The type of split to use.                              |
| **RETURNS** | Tuple[Iterable[Doc], Iterable[Doc]] | The training and testing spaCy Doc objects. |


### Splitters Catalogue

### <kbd>vs_split.splitters</kbd> `wasserstein.v1`

Perform adversarial splitting using a divergence maximization method involving [Wasserstein distance](https://en.wikipedia.org/wiki/Wasserstein_metric).

This method approximates the test split by performing nearest-neighbor search on
a random centroid. Based on Søgaard, Ebert et al.'s work on '[We Need to Talk
About Random Splits](https://aclanthology.org/2021.eacl-main.156/)' (EACL 2021).

| Argument    | Type       | Description                                            |
|-------------|------------|--------------------------------------------------------|
| `*X`        | Iterable   | An iterable of features, preferably a `numpy.ndarray`. |
| `*y`        | Iterable   | An iterable of labels, preferably a `numpy.ndarray`.   |
| `test_size` | float      | The number of neighbors to query. Defaults to `0.2`    |
| `leaf_size` | int        | The leaf size parameter for nearest neighbor search. High values are slower. Defaults to `3`.    |
| **RETURNS** | Tuple[Iterable[Any], Iterable[Any], Iterable[Any], Iterable[Any]] | The training and testing features and labels (i.e. `X_train`, `y_train`, `X_test`, `y_test`). |


### <kbd>vs_split.splitters</kbd> `spacy-wasserstein.v1`

spaCy-compatible version of `wasserstein.v1`. If no vectors were found in the 
`Doc` object, then TF-IDF is computed.

| Argument    | Type         | Description                                            |
|-------------|--------------|--------------------------------------------------------|
| `*docs`     | Iterable[Doc]| An iterable of spaCy Doc objects to split.             |
| `test_size` | float      | The number of neighbors to query. Defaults to `0.2`.    |
| `leaf_size` | int        | The leaf size parameter for nearest neighbor search. High values are slower. Defaults to `3`.    |
| `use_counts`| bool       | Use count vectors instead of initialized vectors. If no vectors were found, the count vectors are automatically used. Defaults to `False`.   | 
| `min_df`    | Union[int, float] | remove terms that appear too infrequently given a threshold. Defaults to `0.10`. | 
| `n_jobs`    | Optional[int]   | Number of parallel jobs to run for neighbor search. Defaults to `-1` (use all CPUs). |
| **RETURNS** | Tuple[Iterable[Doc], Iterable[Doc]] | The training and testing spaCy Doc objects. |


### <kbd>vs_split.splitters</kbd> `doc-length.v1`

Heuristic split based on document length.

By default, it looks for a sentence length threshold, and puts all the longer
sentences in the test split. The threshold is chosen so that approximately 10%
of the data ends up in the test set. 

| Argument    | Type         | Description                                            |
|-------------|--------------|--------------------------------------------------------|
| `*docs`     | Iterable[Doc]| An iterable of spaCy Doc objects to split.             |
| `test_size` | Optional[float]      | The size of the test set for determining the split. Defaults to `0.1`.    |
| `length_threshold` | Optional[int] | Arbitrary length to split the dataset against. Defaults to `None`. |
| **RETURNS** | Tuple[Iterable[Doc], Iterable[Doc]] | The training and testing spaCy Doc objects. |

### <kbd>vs_split.splitters</kbd> `morph-attrs-split.v1`

Perform a heuristic split based on morphological attributes.

This method is loosely-based on the paper: '[(Un)solving Morphological Inflection: Lemma Overlap Artificially Inflates Models' Performance](https://aclanthology.org/2022.acl-short.96/)' by Goldman
et. al (ACL 2022). However, instead of focusing solely on lemma splits, this
method uses morphological attributes. The main motivation is because splitting
on lemma doesn't translate on standard texts.


| Argument    | Type         | Description                                            |
|-------------|--------------|--------------------------------------------------------|
| `*docs`     | Iterable[Doc]| An iterable of spaCy Doc objects to split.             |
| `attrs`     | List[str]     | Morphological attributes to split against. Default is `["Number", "Person"]`.
| `test_size` | Optional[float]      | The size of the test set for determining the split. Defaults to `0.1`.    |
| **RETURNS** | Tuple[Iterable[Doc], Iterable[Doc]] | The training and testing spaCy Doc objects. |


### <kbd>vs_split.splitters</kbd> `entity-switch.v1`

Manually perturb the test set by switching entities based on a given
dictionary of patterns.

This work is based on the paper, '[Entity-Switched Datasets - An Approach to
Auditing the In-Domain Robustness of Named Entity Recognition
Models](https://arxiv.org/abs/2004.04123)' by Agarwal et al. You can control
which entity labels are switched using a **patterns dictionary**.

The patterns dictionary should have **the entity label as the key and a list of
strings as its values.** For example, if we want to switch all `ORG` entities in
the original document with values such as `Bene Gesserit`, `Landsraad`, or
`Spacing Guild`, then we should provide a dictionary that look like this:

```python
patterns = {'ORG': ['Bene Gesserit', 'Landsraad', 'Spacing Guild']}
```

You can add as many patterns or entity labels in the dictionary. The pattern
chosen for substitution is done via
[`random.choice`](https://docs.python.org/3/library/random.html#random.choice).

Note that for `PER` names, this splitter **does not** differentiate between
first or full names. It just performs a drop-in replacement.

| Argument    | Type         | Description                                            |
|-------------|--------------|--------------------------------------------------------|
| `*docs`     | Iterable[Doc]| An iterable of spaCy Doc objects to split.             |
| `*patterns` | Dict[str, List[str]] | Dictionary of patterns for substitution.             |
| `test_size` | Optional[float]      | If provided, then the docs will be split further. Since entity-switching is only needed for the test set, you can just pass the test documents in this function. Defaults to `None`.    |
| **RETURNS** | Tuple[Iterable[Doc], Iterable[Doc]] | The training and testing spaCy Doc objects. |
