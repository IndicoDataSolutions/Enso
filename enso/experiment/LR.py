"""Module for any LR-style experiment."""
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from enso.experiment import Experiment


class GridSearchLR(Experiment):
    """Basic implementation of a grid-search optimized Logistic Regression."""

    def __init__(self):
        """Initialize internal classifier."""
        self.model = LogisticRegression
        self.active_model = None
        self.param_grid = {'penalty': ['l1', 'l2']}

    def train(self, training_data, training_labels):
        """Run grid search to optimize hyper-parameters, then trains the final model."""
        classifier = GridSearchCV(self.model(), self.param_grid)
        classifier.fit(training_data, training_labels)
        # Train with model with ideal params on whole training set
        self.active_model = self.model(**classifier.best_params_)
        self.active_model.fit(training_data, training_labels)

    def predict(self, dataset):
        """Predict results on test set based on current internal model."""
        labels = self.active_model.classes_
        probabilities = self.active_model.predict_proba(dataset)
        return pd.DataFrame({label: probabilities[:, i] for i, label in enumerate(labels)})
