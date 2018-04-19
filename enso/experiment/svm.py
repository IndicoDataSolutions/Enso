from sklearn.svm import SVC
import pandas as pd

from enso.experiment.grid_search import GridSearch


class SupportVectorMachineCV(GridSearch):
    """Implementation of a grid-search optimized RBF-SVM."""

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        self.base_model = SVC
        self.param_grid = {
            'C': [0.01, 0.1, 1.0, 10., 100.],
            'gamma': [0.1, 1.0, 10.],
            'probability': [True]
        }
