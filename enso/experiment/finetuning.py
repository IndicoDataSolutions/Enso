import os

import pandas as pd
from finetune import Classifier

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
        self.model = Classifier(lm_loss_coef=0.)

    def fit(self, X, y):
        """
        :param X: `np.ndarray` of raw text sampled from training data.
        :param y: `np.ndarray` of corresponding targets sampled from training data.
        """
        self.model.fit(X, y)

    def predict(self, X, **kwargs):
        """Predict results on test set based on current internal model."""
        preds = self.model.predict_proba(X)
        return pd.DataFrame.from_records(preds)
