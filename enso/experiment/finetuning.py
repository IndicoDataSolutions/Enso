import os
import json

import pandas as pd
import numpy as np

from indicoio.custom import Collection
from finetune import Classifier, SequenceLabeler

from enso.experiment import ClassificationExperiment
from enso.config import RESULTS_DIRECTORY
from enso.registry import Registry, ModeKeys
try:
    from finetune.base_models.bert.model import BERTModelCased, BERTModelLargeCased, RoBERTa, DistilBERT
except:
    pass

try:
    from finetune.base_models.oscar.model import GPCModel
except:
    pass
try:
    from finetune.base_models import GPT2ModelMedium, GPT2Model
except:
    pass


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class FinetuneDistilBERT(ClassificationExperiment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = Classifier(val_size=0, base_model=DistilBERT, max_length=512)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, **kwargs):
        preds = self.model.predict_proba(X)
        return pd.DataFrame.from_records(preds)

    def cleanup(self):
        del self.model


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class FinetuneRoBERTa(ClassificationExperiment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = Classifier(val_size=0, base_model=RoBERTa, max_length=512)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X, **kwargs):
        preds = self.model.predict_proba(X)
        return pd.DataFrame.from_records(preds)

    def cleanup(self):
        del self.model
                                                                                            

@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class FinetuneGPC(ClassificationExperiment):
    param_grid = {}

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.model = Classifier(val_size=0, base_model=GPCModel, max_length=512)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, **kwargs):
        preds = self.model.predict_proba(X)
        return pd.DataFrame.from_records(preds)

    def cleanup(self):
        del self.model


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class FinetuneGPCPrefit(ClassificationExperiment):
    param_grid = {}

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.model = Classifier(val_size=0, base_model=GPCModel, max_length=256, prefit_init=True)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, **kwargs):
        preds = self.model.predict_proba(X)
        return pd.DataFrame.from_records(preds)

    def cleanup(self):
        del self.model

@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class FinetuneBERT(ClassificationExperiment):
    param_grid = {}

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.model = Classifier(val_size=0, base_model=BERTModelCased, max_length=256)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, **kwargs):
        preds = self.model.predict_proba(X)
        return pd.DataFrame.from_records(preds)

    def cleanup(self):
        del self.model

@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class FinetunGPT2(ClassificationExperiment):
    param_grid = {}

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.model = Classifier(val_size=0, base_model=GPT2Model, max_length=256)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, **kwargs):
        preds = self.model.predict_proba(X)
        return pd.DataFrame.from_records(preds)

    def cleanup(self):
        del self.model

@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class FinetuneBERTLarge(ClassificationExperiment):
    param_grid = {}

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.model = Classifier(val_size=0, base_model=BERTModelLargeCased, max_length=256)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, **kwargs):
        preds = self.model.predict_proba(X)
        return pd.DataFrame.from_records(preds)

    def cleanup(self):
        del self.model

@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class FinetuneGPTSummaries(ClassificationExperiment):
    param_grid = {}

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.model = Classifier(val_size=0, base_model_path="SummariesBase.jl")

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, **kwargs):
        preds = self.model.predict_proba(X)
        return pd.DataFrame.from_records(preds)

    def cleanup(self):
        del self.model

        

@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class FinetuneGPT(ClassificationExperiment):
    """                                                                                                                                                                              
    LanguageModel finetuning as an alternative to simple models trained on top of pretrained features.                                                                               
    """

    param_grid = {}

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.model = Classifier(val_size=0)

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

    def cleanup(self):
        del self.model

@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class FinetuneGPTAdaptors(FinetuneGPT):
    pass

@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class Finetune(ClassificationExperiment):
    """
    LanguageModel finetuning as an alternative to simple models trained on top of pretrained features.
    """

    param_grid = {}

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.model = Classifier(val_size=0)

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

    def cleanup(self):
        del self.model


@Registry.register_experiment(ModeKeys.SEQUENCE, requirements=[("Featurizer", "PlainTextFeaturizer")])
class IndicoSequenceLabel(ClassificationExperiment):

    def __new__(cls, *args, **kwargs):
        raise Exception("DO NOT run this at the moment.... - waiting to hear from Madison")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None

    def fit(self, X, y):
        self.model = Collection("Enso-Sequence-Labeling-{}".format(str(hash(str(X) + str(y)))))
        try:
            self.model.clear()
        except:
            pass

        X = list(zip(X, y))
        for x in X:
            self.model.add_data([x])
        self.model.train()
        self.model.wait()

    def predict(self, X, **kwargs):
        batch_size = 5
        num_samples = len(X)
        num_batches = num_samples // batch_size
        predictions = []
        for i in range(num_batches):
            data = X[i * batch_size: (i + 1) * batch_size]
            predictions.extend(self.model.predict(data))
        return predictions

    def cleanup(self):
        self.model.clear()

@Registry.register_experiment(ModeKeys.SEQUENCE, requirements=[("Featurizer", "PlainTextFeaturizer")])
class FinetuneSequenceLabel(ClassificationExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = SequenceLabeler(val_size=0)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, **kwargs):
        return self.model.predict(X)
