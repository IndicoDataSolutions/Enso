"""Module for any LR-style experiment."""
from sklearn.linear_model import LogisticRegression
import sklearn_crfsuite
import numpy as np
import pandas as pd
import spacy
import crf_processing
from enso.experiment import ClassificationExperiment
from enso.experiment.grid_search import GridSearch
from enso.registry import Registry, ModeKeys


@Registry.register_experiment(
    ModeKeys.RATIONALIZED, requirements=[("Featurizer", "PlainTextFeaturizer")]
)
class CRFLogit(ClassificationExperiment):
    """Implementation of a grid-search optimized Logistic Regression model."""
    NLP = None

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(auto_resample=False, *args, **kwargs)

        if self.NLP is None:
            self.NLP = spacy.load('en_vectors_web_lg')
        self.model = LogisticRegression()
        # Params currently fixed. TODO: sweep
        self.extraction_model = sklearn_crfsuite.CRF(
                algorithm='lbfgs',
                c1=0.1,
                c2=0.1,
                max_iterations=100,
                all_possible_transitions=True
        )

        self.param_grid = {
            "penalty": ["l2"],
            "max_iter": [500],
            "C": [0.1, 1.0, 10.0, 100.0, 1000.0],
            "solver": ["lbfgs"],
            "multi_class": ["multinomial"],
        }
    def spacy_vectorize(self, texts):
        return [i.vector for i in self.NLP.pipe(texts, disable=['parser', 'ner', 'textcat'])]


    def fit(self, X, Y):

        # Make special features for the CRF model
        spans = crf_processing.get_spans_enso(X, Y, trim_front=True)
        spantexts = [crf_processing.get_span_texts(span) for span in spans]
        features = [crf_processing.make_crf_features(spantext) for spantext in spantexts]
        X_train_crf = [crf_processing.sent2features(s) for s in features]
        Y_train_crf = [crf_processing.sent2labels(s) for s in features] 

        self.extraction_model.fit(X_train_crf, Y_train_crf)
        
        X_train_logit = self.spacy_vectorize([crf_processing.crf_filter_text(text, self.extraction_model) for text in X])
        Y_train_logit = [y[1] for y in Y]
        resampled_x, resampled_y = self.resample(X_train_logit, Y_train_logit)
        self.model.fit(resampled_x, resampled_y)
        #self.model.fit(X_train_logit, Y_train_logit)

    def predict(self, X, **kwargs):
        texts = [crf_processing.crf_filter_text(text, self.extraction_model) for text in X]
        X_vectors = self.spacy_vectorize(texts)
        probas = self.model.predict_proba(X_vectors)
        labels = self.model.classes_
        
        return pd.DataFrame({label: probas[:, i] for i, label in enumerate(labels)})
