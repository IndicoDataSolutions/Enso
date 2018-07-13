import os
import json

import pandas as pd

from indicoio.custom import Collection
from finetune import LanguageModelClassifier, LanguageModelSequence
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
        self.model = LanguageModelClassifier(autosave_path=os.path.join(RESULTS_DIRECTORY, '.autosave'))

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
        self.model = LanguageModelSequence(autosave_path=os.path.join(RESULTS_DIRECTORY, '.autosave'))

    def fit(self, X, y):
        X = list(zip(X, [json.loads(y_) for y_ in y]))
        data = indico_to_finetune_sequence(X, "<PAD>")
        self.model.fit(data)

    def predict(self, X, **kwargs):
        predictions = self.model.predict(list(zip(*[X])))
        return [json.dumps(targ[1]) for targ in finetune_to_indico_sequence(predictions, "<PAD>")]


class IndicoSequenceLabel(ClassificationExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = Collection("Enso-Sequence-Labeling")
        try:
            self.model.clear()
        except:
            pass

    def fit(self, X, y):
        X = list(zip(X, [json.loads(y_) for y_ in y]))
        for x in X:
            self.model.add_data([x])
            break
        self.model.train()
        self.model.wait()

    def predict(self, X, **kwargs):
        predictions = self.model.predict(X)
        return [json.dumps(targ) for targ in predictions]
