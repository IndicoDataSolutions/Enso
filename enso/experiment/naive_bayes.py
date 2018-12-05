from sklearn.naive_bayes import GaussianNB, MultinomialNB
import pandas as pd

from enso.experiment import Experiment

from enso.registry import Registry, ModeKeys


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "not PlainTextFeaturizer")])
class NaiveBayes(Experiment):
    """Gaussian naive bayes model."""

    param_grid = {}

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.model = GaussianNB()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, **kwargs):
        """Predict results on test set based on current internal model."""
        labels = self.model.classes_
        probabilities = self.model.predict_proba(X)
        return pd.DataFrame({label: probabilities[:, i] for i, label in enumerate(labels)})
