# ⚔️ vs-split: a library for creating adversarial splits

Have you ever encountered a problem where **your model works well in your test set
but doesn't perform well in the wild?**  It's likely because your test set does
not reflect the reality of your domain, overestimating your model's performance.[^1]

This library provides **alternative ways to split and sanity-check your datasets**
and ensure they're robust once you deploy them into production.

[^1]: Check out my blog post, [*Your train-test split may be doing you a disservice*](https://ljvmiranda921.github.io/2022/08/30/adversarial-splits/), for a technical overview of this problem.

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

The library exposes two main functions: 

- **`train_test_split(X: Iterable, y: Iterable, split_id: str, **attrs)`** that accepts [NumPy arrays](https://numpy.org/doc/stable/reference/generated/numpy.array.html) of your features and labels; and&mdash; 
- **`spacy_train_test_split(docs: Iterable[Doc], split_id: str, **attrs)`** that accepts an iterable of [spaCy Doc objects](https://spacy.io/api/doc).[^2] spaCy is a Python library for natural language processing, and the Doc object is one of its core data structures. This function is useful if you're working on linguistic data.  

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

You can also register custom splitters via the `splitters` catalogue:

```python
import random
from vs_split.splitters import splitters

@splitters.register("random-spacy.v1")
def random_spacy(docs, train_size:):
    pass

```




### More examples


## API


### Splitters Catalogue
