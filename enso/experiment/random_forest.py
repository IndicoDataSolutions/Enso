from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from enso.experiment.grid_search import GridSearch
from enso.registry import Registry, ModeKeys


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "not PlainTextFeaturizer")])
class RandomForestCV(GridSearch):
    """Implementation of a grid-search optimized RandomForest."""

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.base_model = RandomForestClassifier
        self.param_grid = {
            'n_estimators': [10, 50]
        }
