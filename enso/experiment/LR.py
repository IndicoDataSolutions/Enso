"""Module for any LR-style experiment."""
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from enso.experiment import Experiment


class GridSearchLR(Experiment):
    """Basic implementation of a grid-search optimized Logistic Regression."""

    param_grid = {'penalty': ['l2'], 'C': [0.1, 1.0, 10., 100., 1000.], 'multi_class': 'multinomial'}

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.model = LogisticRegression
        self.active_model = None

    def train(self, training_data, training_labels):
        """Run grid search to optimize hyper-parameters, then trains the final model."""
        classifier = GridSearchCV(self.model(), self.param_grid)
        classifier.fit(training_data, training_labels)
        # Train with model with ideal params on whole training set
        self.active_model = self.model(**classifier.best_params_)
        self.active_model.fit(training_data, training_labels)

    def predict(self, dataset, **kwargs):
        """Predict results on test set based on current internal model."""
        labels = self.active_model.classes_
        probabilities = self.active_model.predict_proba(dataset)
        return pd.DataFrame({label: probabilities[:, i] for i, label in enumerate(labels)})


class LR(Experiment):
    """Basic implementation of a grid-search optimized Logistic Regression."""

    param_grid = {'penalty': ['l2'], 'C': [100., 10., 1.], 'multi_class': ['multinomial'], 'solver': ['lbfgs']}

    def __init__(self, penalty, C, multi_class, solver, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.model = LogisticRegression(
            penalty=penalty,
            C=C,
            multi_class=multi_class,
            solver=solver
        )

    def train(self, training_data, training_labels):
        """Run grid search to optimize hyper-parameters, then trains the final model."""
        self.model.fit(training_data, training_labels)

    def predict(self, dataset, **kwargs):
        """Predict results on test set based on current internal model."""
        labels = self.model.classes_
        probabilities = self.model.predict_proba(dataset)
        return pd.DataFrame({label: probabilities[:, i] for i, label in enumerate(labels)})
