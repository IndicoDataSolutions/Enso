"""Module for any tfidf experiment."""

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from enso.experiment.grid_search import GridSearch

from enso.registry import Registry, ModeKeys

@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class TfidfLogisticRegression(GridSearch):
    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.base_model = LogisticRegression
        self.param_grid = {
            'penalty': ['l1'],
            'C': [0.01, 0.1, 1.0, 10., 100., 1000.],
        }
        self.tfidf = TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english',
                                     ngram_range=(1, 3), dtype=np.float32)

    def fit(self, X, y):
        self.tfidf.fit(list(set(X)))
        X_p = self.tfidf.transform(X).todense()
        super().fit(X_p, y)

    def predict(self, X, **kwargs):
        """Predict results on test set based on current internal model."""
        X_p = self.tfidf.transform(X)
        return super().predict(X_p)
