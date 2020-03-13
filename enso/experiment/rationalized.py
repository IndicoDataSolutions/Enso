import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import spacy
from enso.experiment import ClassificationExperiment
from sklearn.linear_model import LogisticRegression
from enso.utils import OversampledKFold
from enso.registry import Registry, ModeKeys
from enso.experiment.grid_search import GridSearch
from finetune import Classifier, SequenceLabeler
from sklearn.preprocessing import LabelBinarizer
from collections import Counter, defaultdict

class RationalizedGridSearch(GridSearch):
    def fit(self, X, y):
        return super().fit(X, [yi[1] for yi in y])


@Registry.register_experiment(ModeKeys.RATIONALIZED, requirements=[("Featurizer", "not PlainTextFeaturizer")])
class LRBaselineNonRationalized(RationalizedGridSearch):
    """Implementation of a grid-search optimized Logistic Regression model."""
    
    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.base_model = LogisticRegression
        self.param_grid = {
            "penalty": ["l2"],
            "max_iter": [500],
            "C": [0.1, 1.0, 10.0, 100.0, 1000.0],
            "solver": ["lbfgs"],
            "multi_class": ["multinomial"],
        }

        
@Registry.register_experiment(ModeKeys.RATIONALIZED, requirements=[("Featurizer", "PlainTextFeaturizer")])
class FinetuneClfBaselineNonRationalized(ClassificationExperiment):
    param_grid = {}
    
    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(auto_resample=False, *args, **kwargs)
        self.model = Classifier(val_size=0)
        
    def fit(self, X, y):
        self.model.fit(*self.resample(X, [yi[1] for yi in y]))
        
    def predict(self, X, **kwargs):
        preds = self.model.predict_proba(X)
        return pd.DataFrame.from_records(preds)

    def cleanup(self):
        del self.model

        
def safe_mean(l):
    if l:
        return sum(l) / len(l)
    return 0
        
        
@Registry.register_experiment(ModeKeys.RATIONALIZED, requirements=[("Featurizer", "PlainTextFeaturizer")])
class FinetuneSeqBaselineRationalized(ClassificationExperiment):
    param_grid = {}

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(auto_resample=False, *args, **kwargs)
        self.model = SequenceLabeler(val_size=0)

    def fit(self, X, y):
        targets = []
        for x, l in zip(X, y):
            if l[0]:
                targets.append([{**label, "label": l[1]} for label in l[0]])
            else:
                targets.append([{"start": 0, "end": len(x), "label": l[1], "text": x}])
        idxs, _ = self.resample(list(range(len(X))), [yi[1] for yi in y])
        train_x = []
        train_y = []
        for i in idxs:
            train_x.append(X[i])
            train_y.append(targets[i])
        self.model.fit(train_x, train_y)

    def predict(self, X, **kwargs):
        preds = self.model.predict_proba(X)
        classes = self.model.input_pipeline.label_encoder.classes_[:]
        classes.remove("<PAD>")
        output = []
        
        for sample in preds:
            output.append(
                {k: safe_mean([s["confidence"][k] for s in sample]) + 1e-10 for k in classes}
            )
        return pd.DataFrame.from_records(output)

    def cleanup(self):
        del self.model


@Registry.register_experiment(ModeKeys.RATIONALIZED, requirements=[("Featurizer", "PlainTextFeaturizer")])
class ReweightedGloveClassifier(ClassificationExperiment):

    NLP = None

    def __init__(self, *args, **kwargs):
        super().__init__(auto_resample=False, *args, **kwargs)
        self.model = LogisticRegression()
        self.p_rationale_given_word = {}
        if self.NLP is None:
            self.NLP = spacy.load('en_vectors_web_lg')
   
    def _compute_p_rationale(self, docs, rationale_docs):
        word_counts = Counter([
            token.text.lower() for doc in docs for token in doc
        ])
        rationale_word_counts = Counter([
            token.text.lower() for doc in rationale_docs for token in doc
        ])

        # smoothing for unseen terms
        base_freq = 1. / sum(word_counts.values())
        self.p_rationale_given_word = defaultdict(lambda: base_freq)

        for word, count in rationale_word_counts.items():
            # there can be slight difference in tokenization -- so we set 
            # word counts to be the rationale word count when we encounter this
            self.p_rationale_given_word[word] = count / word_counts.get(word, count)

        return self.p_rationale_given_word

    def _featurize(self, doc):
        doc_vect = np.mean([
            token.vector * self.p_rationale_given_word[token.text.lower()]
            for token in doc
        ], axis=0)
        normed_doc_vect = doc_vect / np.linalg.norm(doc_vect)
        return normed_doc_vect

    def fit(self, X, Y):
        # In this naive scenario we're only trying to downweight
        # irrelevant terms -- rationales are shared across classes
        rationales = []
        labels = []
        for x, l in zip(X, Y):
            labels.append(l[1])
            if l[0]:
                rationales.append([{**label, "label": l[1]} for label in l[0]])
            else:
                rationales.append([{"start": 0, "end": len(x), "label": l[1], "text": x}])
        rationale_texts = [
            rationale['text'] 
            for doc in rationales 
            for rationale in doc
        ]
        docs = np.asarray([self.NLP(str(x), disable=['ner', 'tagger', 'textcat']) for x in X])
        rationale_docs = np.asarray([self.NLP(rationale) for rationale in rationale_texts])
        self._compute_p_rationale(docs, rationale_docs)

        doc_vects = np.asarray([self._featurize(doc) for doc in docs])
        resampled_x, resampled_y = self.resample(doc_vects, labels)
        self.model.fit(resampled_x, resampled_y)


    def predict(self, X, **kwargs):
        docs = np.asarray([self.NLP(str(x), disable=['ner', 'tagger', 'textcat']) for x in X])
        doc_vects = np.asarray([self._featurize(doc) for doc in docs])
        probas = self.model.predict_proba(doc_vects) 
        labels = self.model.classes_
        
        return pd.DataFrame(
            {label: probas[:, i] for i, label in enumerate(labels)}
        )
      
