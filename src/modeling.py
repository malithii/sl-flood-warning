"""Helpers for model training."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.metaestimators import available_if


def _inner_has(attr):
    return lambda self: hasattr(self.estimator, attr)


class NegativeSubsampler(ClassifierMixin, BaseEstimator):
    """Fit `estimator` on all positive rows plus a random sample of negative rows.

    KNN and RBF-SVM scale badly with ~200,000 training rows. With only a few hundred
    positives, keeping every positive and 10,000 negatives keeps nearly all the information
    while making these models fast enough to tune. Scores are only used for ranking and
    thresholding, so the changed class ratio does not matter.
    """

    def __init__(self, estimator, n_negatives=10_000, random_state=42):
        self.estimator = estimator
        self.n_negatives = n_negatives
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        rng = np.random.RandomState(self.random_state)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        if len(neg) > self.n_negatives:
            neg = rng.choice(neg, self.n_negatives, replace=False)
        idx = np.sort(np.r_[pos, neg])
        Xs = X.iloc[idx] if hasattr(X, "iloc") else np.asarray(X)[idx]
        self.estimator_ = clone(self.estimator).fit(Xs, y[idx])
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    @available_if(_inner_has("predict_proba"))
    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    @available_if(_inner_has("decision_function"))
    def decision_function(self, X):
        return self.estimator_.decision_function(X)
