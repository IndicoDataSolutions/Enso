from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from enso.experiment import Experiment


class RandomForest(Experiment):
    """Basic implementation of a grid-search optimized Logistic Regression."""

    param_grid = {'n_estimators': [10, 50]}

    def __init__(self, n_estimators, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.model = RandomForestClassifier(n_estimators=n_estimators)

    def train(self, training_data, training_labels):
        """Run grid search to optimize hyper-parameters, then trains the final model."""
        self.model.fit(training_data, training_labels)

    def predict(self, dataset, **kwargs):
        """Predict results on test set based on current internal model."""
        labels = self.model.classes_
        probabilities = self.model.predict_proba(dataset)
        return pd.DataFrame({label: probabilities[:, i] for i, label in enumerate(labels)})
