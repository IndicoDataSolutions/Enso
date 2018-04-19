from sklearn.naive_bayes import GaussianNB, MultinomialNB
import pandas as pd

from enso.experiment import Experiment


class NaiveBayes(Experiment):
    """Basic implementation of a grid-search optimized Logistic Regression."""

    param_grid = {}

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.model = GaussianNB()

    def fit(self, X, y):
        """Run grid search to optimize hyper-parameters, then trains the final model."""
        self.model.fit(X, y)

    def predict(self, X, **kwargs):
        """Predict results on test set based on current internal model."""
        labels = self.model.classes_
        probabilities = self.model.predict_proba(X)
        return pd.DataFrame({label: probabilities[:, i] for i, label in enumerate(labels)})
