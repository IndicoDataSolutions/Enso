import os
import json

import pandas as pd

from indicoio.custom import Collection
from finetune import Classifier, SequenceLabeler
from finetune.utils import finetune_to_indico_sequence, indico_to_finetune_sequence

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
        self.model = Classifier(autosave_path=os.path.join(RESULTS_DIRECTORY, '.autosave'))

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


class FinetuneSequenceLabel(ClassificationExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = SequenceLabeler(autosave_path=os.path.join(RESULTS_DIRECTORY, '.autosave'))

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, **kwargs):
        predictions = self.model.predict(list(zip(*[X])))
        return [json.dumps(targ) for targ in predictions]


class IndicoSequenceLabel(ClassificationExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None

    def fit(self, X, y):
        self.model = Collection("Enso-Sequence-Labeling-{}".format(str(hash(str(X)+str(y)))))
        try:
            self.model.clear()
        except:
            pass
        X = list(zip(X, [json.loads(y_) for y_ in y]))
        for x in X:
            self.model.add_data([x])
        self.model.train()
        self.model.wait()

    def predict(self, X, **kwargs):
        predictions = self.model.predict(X)
        return [json.dumps(targ) for targ in predictions]

    def __del__(self):
        self.model.clear()

