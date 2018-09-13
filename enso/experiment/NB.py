"""Module for any NB-style experiment."""
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from enso.experiment import Experiment

from enso.registry import Registry, ModeKeys


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "not PlainTextFeaturizer")])
class NaiveGaussianBayes(Experiment):
    """Basic implementation of a grid-search optimized Logistic Regression."""

    def __init__(self):
        """Initialize internal classifier."""
        self.model = GaussianNB
        self.active_model = None

    def train(self, training_data, training_labels):
        """Run grid search to optimize hyper-parameters, then trains the final model."""
        self.active_model = self.model()
        self.active_model.fit(training_data, training_labels)

    def predict(self, dataset):
        """Predict results on test set based on current internal model."""
        labels = self.active_model.classes_
        probabilities = self.active_model.predict_proba(dataset)
        return pd.DataFrame({label: probabilities[:, i] for i, label in enumerate(labels)})


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "not PlainTextFeaturizer")])
class NaiveMultinomialBayes(Experiment):
    """Basic implementation of a grid-search optimized Logistic Regression."""

    def __init__(self):
        """Initialize internal classifier."""
        self.model = MultinomialNB
        self.active_model = None
        self.param_grid = {'alpha': [0, 0.25, 0.5, 0.75, 1], 'fit_prior': [True, False]}

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


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "not PlainTextFeaturizer")])
class NativeBernoulliBayes(Experiment):
    """Basic implementation of a grid-search optimized Logistic Regression."""

    def __init__(self):
        """Initialize internal classifier."""
        self.model = BernoulliNB
        self.active_model = None
        self.param_grid = {'alpha': [0, 0.25, 0.5, 0.75, 1], 'fit_prior': [True, False], 'binarize': [True, False]}

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
