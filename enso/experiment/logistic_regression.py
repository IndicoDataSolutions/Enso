"""Module for any LR-style experiment."""
from sklearn.linear_model import LogisticRegression

from enso.experiment.grid_search import GridSearch
from enso.registry import Registry, ModeKeys


@Registry.register_experiment(
    ModeKeys.CLASSIFY, requirements=[("Featurizer", "not PlainTextFeaturizer")]
)
class LogisticRegressionCV(GridSearch):
    """Implementation of a grid-search optimized Logistic Regression model."""

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.base_model = LogisticRegression
        self.param_grid = {
            "penalty": ["l2"],
            "max_iter": [500],
            "C": [0.1, 1.0, 10.0, 100.0, 1000.0],
            "solver": ["lbfgs"],
            "multi_class": ["multinomial"],
        }
