# vs-split (WIP)

A Python library for creating adversarial splits.


```python
from vs_split import train_test_split

X_train, y_train, X_test, y_test = train_test_split(X_data, y_data, split="wasserstein")
```

The `vs-split` library might look like it has a similar API with scikit-learn's
`train_test_split`, but it does not.  Unlike the latter,
`vs_split.standard.train_test_split` doesn't expect an arbitrary number of
iterables, and the keyword parameters are different too.