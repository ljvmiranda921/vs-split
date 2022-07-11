# ⚔️ vs-split: a library for creating adversarial splits

Have you ever encountered a problem where **your model works well in your test set
but doesn't perform well in the wild?**  It's likely because your test set does
not reflect the reality of your domain, overestimating your model's performance.

This library provides **alternative ways to split and sanity-check your datasets**,
and ensure that they're robust once you deploy them into production.

## Install

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

## Usage

The `vs-split` library exposes two main functions: (1) **`train_test_split`** that
accepts [NumPy
arrays](https://numpy.org/doc/stable/reference/generated/numpy.array.html) of
your features and labels, and (2) **`spacy_train_test_split`** that accepts a list
of [spaCy Doc objects](https://spacy.io/api/doc).  For both functions, you can
provide the type of split in the `split_id` parameter (c.f. [Splitters
Catalogue](#splitters-catalogue)) and pass custom keyword-arguments.

```python
from vs_split import train_test_split, spacy_train_test_split

# For most datasets
X_train, y_train, X_test, y_test = train_test_split(X_data, y_data, split_id="wasserstein.v1")
# For spaCy Doc objects
docs_train, docs_test = spacy_train_test_split(docs, split_id="wasserstein-spacy.v1")
```

The `vs-split` library might look like it has a similar API with [scikit-learn's
`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html),
but it does not.  Unlike the latter, `vs_split.train_test_split` doesn't expect
an arbitrary number of iterables, and the keyword parameters are different too.

### Registering your own splitters


### More examples


## API


### Splitters Catalogue