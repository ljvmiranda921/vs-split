# vs-split

A Python library for creating adversarial splits.

The `vs-split` library is similar to scikit-learn's `train_test_split`!

```python
from vs_split import train_test_split

X_train, y_train, X_test, y_test = train_test_split(X_data, y_data, split="wasserstein")
```