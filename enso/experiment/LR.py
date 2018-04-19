"""Module for any LR-style experiment."""
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from enso.experiment.grid_search import GridSearch


class LogisticRegressionCV(GridSearch):
    """Basic implementation of a grid-search optimized Logistic Regression."""



    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.base_model = LogisticRegression
        self.param_grid = {
            'penalty': ['l2'],
            'C': [0.1, 1.0, 10., 100., 1000.],
        }
