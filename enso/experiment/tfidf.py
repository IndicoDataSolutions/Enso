"""Module for any tfidf experiment."""

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from enso.experiment import ClassificationExperiment
from enso.experiment.k_centers import KCentersAlgorithm
from enso.experiment.grid_search import GridSearch

from enso.registry import Registry, ModeKeys


class TfidfModel(ClassificationExperiment):
    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.base_model = None
        self.param_grid = {}
        self.tfidf = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            stop_words="english",
            ngram_range=(1, 3),
            dtype=np.float32,
        )

    def fit(self, X, y):
        self.tfidf.fit(list(set(X)))
        X_p = self.tfidf.transform(X).todense()
        super().fit(X_p, y)

    def predict(self, X, **kwargs):
        """Predict results on test set based on current internal model."""
        X_p = self.tfidf.transform(X)
        return super().predict(X_p)


class TfidfModelGridSearch(TfidfModel, GridSearch):
    pass


@Registry.register_experiment(
    ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")]
)
class TfidfLogisticRegression(TfidfModelGridSearch):
    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.base_model = LogisticRegression
        self.param_grid = {
            "solver": ["lbfgs"],
            "multi_class": ["multinomial"],
            "penalty": ["l2"],
            "max_iter": [500],
            "C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        }


@Registry.register_experiment(
    ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")]
)
class TfidfKNN(TfidfModelGridSearch):
    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.base_model = KNeighborsClassifier
        self.param_grid = {
            "metric": ["minkowski", "cosine"],
            "n_neighbors": [1, 2, 4, 8, 16],
            "algorithm": ["brute"],
        }


@Registry.register_experiment(
    ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")]
)
class TfidfKCenters(TfidfModelGridSearch):
    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.base_model = KCentersAlgorithm
        self.param_grid = {"metric": ["cosine", "euclidean", "l1"]}
