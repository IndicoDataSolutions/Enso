from abc import abstractmethod

from sklearn.model_selection import GridSearchCV
import pandas as pd

from enso.experiment import ClassificationExperiment


class GridSearch(ClassificationExperiment):
    """
    Base class for classification models that select hyperparameters via cross validation.
    Assumes the `base_model` property set on child classes inherits from
    `sklearn.base.BaseEstimator` and implements `predict_proba` and `score`.

    :cvar base_model: Class name of base model, must be set by child classes.
    :cvar param_grid: Dictionary that maps from class paramaters to an array of values to search over.  Must be set by child classes.
    """

    param_grid = {}
    base_model = None

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.best_model = None

    def fit(self, X, y):
        """
        Runs grid search over `self.param_grid` on `self.base_model` to optimize hyper-parameters using
        KFolds cross-validation, then retrains using the selected parameters on the full training set.

        :param X: `np.ndarray` of input features sampled from training data.
        :param y: `np.ndarray` of corresponding targets sampled from training data.
        """
        classifier = GridSearchCV(
            self.base_model(),
            param_grid=self.param_grid
        )
        classifier.fit(X, y)

        self.best_model = self.base_model(**classifier.best_params_)
        self.best_model.fit(X, y)

    def predict(self, X, **kwargs):
        """Predict results on test set based on current internal model."""
        labels = self.best_model.classes_
        probabilities = self.best_model.predict_proba(X)
        return pd.DataFrame({label: probabilities[:, i] for i, label in enumerate(labels)})
