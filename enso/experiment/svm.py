from sklearn.svm import SVC
import pandas as pd

from enso.experiment.grid_search import GridSearch
from enso.registry import Registry, ModeKeys


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "not PlainTextFeaturizer"), ("Resampler", "not RandomOverSampler")])
class SupportVectorMachineCV(GridSearch):
    """Implementation of a grid-search optimized RBF-SVM."""

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.base_model = SVC
        self.param_grid = {
            'C': [0.01, 0.1, 1.0, 10., 100.],
            'gamma': [0.1, 1.0, 10.],
            'probability': [True],
            'class_weight': ["balanced", None]
        }
