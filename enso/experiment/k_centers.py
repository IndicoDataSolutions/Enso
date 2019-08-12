from sklearn.metrics import pairwise_distances, accuracy_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator

import pandas as pd
import numpy as np

from collections import Counter
from enso.experiment.grid_search import GridSearch
from enso.registry import Registry, ModeKeys


class KCentersAlgorithm(BaseEstimator):
    def __init__(self, metric="cosine", mean="additive"):
        self.metric = metric
        self.mean = mean
        self.been_fit = False

    def predict(self, X):
        assert self.been_fit
        distances = pairwise_distances(X, self.centers_, metric=self.metric)
        closest_center_idxs = np.argmin(distances, axis=1)
        predictions = self.classes_[closest_center_idxs]
        return predictions

    def score(self, X, Y, sample_weight=None):
        return accuracy_score(Y, self.predict(X))

    def fit(self, X, Y):
        X, Y = check_X_y(X, Y)
        X, Y = np.asarray(X), np.asarray(Y)
        class_counts = Counter(Y)
        self.classes_ = list(class_counts.keys())
        self.centers_ = []
        for label in self.classes_:
            all_samples = np.where(Y == label)
            data = X[all_samples]
            assert len(data) >= 1
            if self.mean == "additive":
                center = np.mean(data, axis=0)
            elif self.mean == "geometric":
                if np.all(np.abs(data) == data):
                    center = np.ones_like(data[0])
                    for example in data:
                        assert np.shape(example) == np.shape(center)
                        nth_root = 1 / len(data)
                        root = np.float_power(example + 1e-7, nth_root)
                        center = np.multiply(center, root)
                else:
                    center = np.mean(data, axis=0)
            else:
                raise ValueError(
                    "Invalid 'mean' configuration {} for KCenters: must be one of ['additive','geometric'].".format(
                        self.metric
                    )
                )
            assert len(np.shape(center)) == 1 and np.shape(center)[0] == len(data[0])
            self.centers_.append(center)

        self.centers_ = np.asarray(self.centers_)
        assert np.shape(self.centers_) == (len(self.classes_), len(data[0]))
        self.classes_ = np.asarray(self.classes_)
        self.been_fit = True
        return self

    def predict_proba(self, X):
        num_examples = np.shape(X)[0]
        num_classes = len(self.classes_)
        distances = pairwise_distances(X, self.centers_, metric=self.metric)
        inverse_distances = 1.0 / (distances + 1e-7)
        inverse_distances = np.square(inverse_distances)
        l1_norms = np.expand_dims(inverse_distances.sum(axis=1), axis=-1)
        probs = (
            inverse_distances / l1_norms
        )  # norm to one to form proper probability distribution
        assert np.shape(probs) == (num_examples, num_classes)

        probs = np.zeros((num_examples, num_classes))
        labels = self.predict(X)
        for i, label in enumerate(labels):
            idx = np.where(self.classes_ == label)
            print(idx)
            probs[i][idx] = 1
        return probs


@Registry.register_experiment(
    ModeKeys.CLASSIFY, requirements=[("Featurizer", "not PlainTextFeaturizer")]
)
class KCenters(GridSearch):
    """Implementation of a grid-search optimized KCenters."""

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.base_model = KCentersAlgorithm
        self.param_grid = {
            "metric": ["cosine", "euclidean", "l1"],
            "mean": ["additive", "geometric"],
        }
