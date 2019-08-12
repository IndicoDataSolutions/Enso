from sklearn.neighbors import KNeighborsClassifier, DistanceMetric

from enso.experiment.grid_search import GridSearch
from enso.registry import Registry, ModeKeys


@Registry.register_experiment(
    ModeKeys.CLASSIFY, requirements=[("Featurizer", "not PlainTextFeaturizer")]
)
class KNNCV(GridSearch):
    """Implementation of a grid-search optimized KNN model."""

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.base_model = KNeighborsClassifier
        self.param_grid = {
            "metric": ["minkowski", "cosine"],
            "n_neighbors": [1, 2, 4, 8, 16],
        }
