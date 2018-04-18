from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd

from enso.experiment import Experiment


class RBFSVM(Experiment):
    """Basic implementation of a grid-search optimized Logistic Regression."""

    param_grid = {'C': [0.01, 0.1, 1.0, 10., 100.], 'gamma': [0.1, 1.0, 10.], 'probability': [True]}

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        self.model = SVC
        self.active_model = None

    def train(self, training_data, training_labels):
        """Run grid search to optimize hyper-parameters, then trains the final model."""
        classifier = GridSearchCV(self.model(), self.param_grid)
        classifier.fit(training_data, training_labels)

        self.active_model = self.model(**classifier.best_params_)
        self.active_model.fit(training_data, training_labels)

    def predict(self, dataset, **kwargs):
        """Predict results on test set based on current internal model."""
        labels = self.active_model.classes_
        probabilities = self.active_model.predict_proba(dataset)
        return pd.DataFrame({label: probabilities[:, i] for i, label in enumerate(labels)})
