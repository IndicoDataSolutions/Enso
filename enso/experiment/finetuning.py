import os

from finetune import LanguageModelClassifier
import pandas as pd

from enso.experiment import ClassificationExperiment
from enso.config import RESULTS_DIRECTORY


class Finetune(ClassificationExperiment):
    """
    LanguageModel finetuning as an alternative to simple models trained on top of pretrained features.
    """

    param_grid = {}

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.model = LanguageModelClassifier(autosave_path=os.path.join(RESULTS_DIRECTORY, '.autosave'))

    def fit(self, X, y):
        """
        Runs grid search over `self.param_grid` on `self.base_model` to optimize hyper-parameters using
        KFolds cross-validation, then retrains using the selected parameters on the full training set.

        :param X: `np.ndarray` of input features sampled from training data.
        :param y: `np.ndarray` of corresponding targets sampled from training data.
        """
        self.model.fit(X, y)

    def predict(self, X, **kwargs):
        """Predict results on test set based on current internal model."""
        preds = self.model.predict_proba(X)
        return pd.DataFrame.from_records(preds)
