from abc import abstractmethod

from sklearn.model_selection import GridSearchCV
import pandas as pd

from enso.experiment import ClassificationExperiment


class GridSearch(ClassificationExperiment):
    """
    Base class for classification models that select hyperparameters via cross validation.
    Assumes the `base_model` property set on child classes inherits from
    `sklearn.base.BaseEstimator` and implements `predict_proba`.
    """

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.param_grid = {}
        self.base_model = None
        self.best_model = None

    def train(self, training_data, training_labels):
        """Run grid search to optimize hyper-parameters, then trains the final model."""
        classifier = GridSearchCV(
            self.base_model(),
            param_grid=self.param_grid
        )
        classifier.fit(training_data, training_labels)

        self.best_model = self.base_model(**classifier.best_params_)
        self.best_model.fit(training_data, training_labels)

    def predict(self, dataset, **kwargs):
        """Predict results on test set based on current internal model."""
        labels = self.best_model.classes_
        probabilities = self.best_model.predict_proba(dataset)
        return pd.DataFrame({label: probabilities[:, i] for i, label in enumerate(labels)})
