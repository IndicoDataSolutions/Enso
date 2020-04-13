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
    """
    Weights words by their proportional occurrence as rationales, smoothed

    """
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
        base_freq = np.sqrt(1. / sum(word_counts.values()))
        self.p_rationale_given_word = defaultdict(lambda: base_freq)

        for word, count in rationale_word_counts.items():
            # there can be slight difference in tokenization -- so we set 
            # word counts to be the rationale word count when we encounter this
            self.p_rationale_given_word[word] = np.sqrt(count / word_counts.get(word, count))

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


class BaseRationaleGridSearch(GridSearch):
    
    NLP = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model = LogisticRegression
        self.param_grid = {
            "penalty": ["l2"],
            "max_iter": [500],
            "C": [0.1, 1.0, 10.0, 100.0, 1000.0],
            "solver": ["lbfgs"],
            "multi_class": ["multinomial"],
        }
        self.model = LogisticRegression()
        self.rationale_weight = {}
        if self.NLP is None:
            self.NLP = spacy.load('en_vectors_web_lg')

    def _valid(self, token):
        return (
            token.has_vector 
            and not token.is_stop 
            and np.any(np.nonzero(token.vector))
        )

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
                rationales.append([])
        rationale_texts = [rationale["text"] for doc in rationales for rationale in doc]
        docs = np.asarray([self.NLP(str(x), disable=["ner", "tagger", "textcat"]) for x in X])
        rationale_docs = np.asarray([self.NLP(rationale) if len(rationale) else None for rationale in rationale_texts])
        self._train_rationale_model(docs, rationale_docs, labels=labels)

        doc_vects = np.asarray([self._featurize(doc) for doc in docs])
        resampled_x, resampled_y = self.resample(doc_vects, labels)
        super().fit(resampled_x, resampled_y)

    def predict(self, X, **kwargs):
        docs = np.asarray([self.NLP(str(x), disable=['ner', 'tagger', 'textcat']) for x in X])
        doc_vects = np.asarray([self._featurize(doc) for doc in docs])
        probas = self.best_model.predict_proba(doc_vects) 
        labels = self.best_model.classes_
        return pd.DataFrame(
            {label: probas[:, i] for i, label in enumerate(labels)}
        )


@Registry.register_experiment(ModeKeys.RATIONALIZED, requirements=[("Featurizer", "PlainTextFeaturizer")])
class DistReweightedGloveClassifierCV(BaseRationaleGridSearch):
    """
    Weights words by cosine similarity to the mean of the rationale vector representations

    """
    def _train_rationale_model(self, docs, rationale_docs, labels=None):
        rationale_vecs = [
            doc.vector / np.linalg.norm(doc.vector)
            for doc in rationale_docs
            if doc and doc.has_vector and np.any(np.nonzero(doc.vector))
        ]
        rationale_proto = np.mean(rationale_vecs, axis=0)
        self.normalized_rationale_proto = rationale_proto / np.linalg.norm(rationale_proto)
               
    def _rationale_weight(self, word):
        cosine_sim = np.dot(word.vector / np.linalg.norm(word.vector), self.normalized_rationale_proto)
        return (1 + cosine_sim) / 2.

    def _featurize(self, doc):
        doc_vect = np.mean([
            token.vector * self._rationale_weight(token)
            for token in doc
            if self._valid(token)
        ], axis=0)

        if not np.any(np.nonzero(doc_vect)):
            return np.zeros_like(doc.vector)
        else:
            return doc_vect / np.linalg.norm(doc_vect)


@Registry.register_experiment(ModeKeys.RATIONALIZED, requirements=[("Featurizer", "PlainTextFeaturizer")])
class DistReweightedGloveByClassClassifierCV(BaseRationaleGridSearch):
    """
    Weights words by cosine similarity to the mean of the rationale vector representations per class

    """
    def _train_rationale_model(self, docs, rationale_docs, labels=None):
        rationale_vecs_by_class = defaultdict(list)
        for doc, label in zip(rationale_docs, labels):
            if doc and doc.has_vector and np.any(np.nonzero(doc.vector)):
                rationale_vecs_by_class[label].append(
                    doc.vector / np.linalg.norm(doc.vector)
                )
        rationale_proto_by_class = {
            label: np.mean(rationale_vecs, axis=0)
            for label, rationale_vecs in rationale_vecs_by_class.items()
        }
        self.normalized_rationale_proto_by_class = OrderedDict({
            label: rationale_proto / np.linalg.norm(rationale_proto)
            for label, rationale_proto in rationale_proto_by_class.items()
        })

    def _rationale_weight(self, word, rationale_proto):
        cosine_sim = np.dot(word.vector / np.linalg.norm(word.vector), rationale_proto)
        return cosine_sim

    def _featurize(self, doc):
        """
        Take the mean representation, reweighted by the representations of
        each of the rationale prototypes

        """
        doc_vects = []
        for rationale_proto in self.normalized_rationale_proto_by_class.values():
            doc_vects.append(
                np.mean(
                    [
                        token.vector * self._rationale_weight(token, rationale_proto)
                        for token in doc if self._valid(token)
                    ],
                    axis=0
                )
            )
        doc_vect = np.mean(doc_vects, axis=0)

        return doc_vect / np.linalg.norm(doc_vect)


@Registry.register_experiment(ModeKeys.RATIONALIZED, requirements=[("Featurizer", "PlainTextFeaturizer")])
class RationaleInformedLRCV(BaseRationaleGridSearch):
    """
    Reweight document vectors by their similarity to a rationale vector, predicted by an LR model
    """
    def _train_rationale_model(self, docs, rationale_docs, labels=None):
        rationale_vecs = [
            doc.vector / np.linalg.norm(doc.vector) 
            for doc in rationale_docs 
            if doc.has_vector and np.any(np.nonzero(doc.vector))
        ]
        rationale_targets = [1] * len(rationale_vecs)
        background_vecs = [
            doc.vector / np.linalg.norm(doc.vector)
            for doc in docs
            if doc.has_vector and np.any(np.nonzero(doc.vector))
        ]
        background_targets = [0] * len(background_vecs)
        X = rationale_vecs + background_vecs
        Y = rationale_targets + background_targets

        cv_rationale_model = GridSearchCV(
            self.base_model(),
            param_grid=self.param_grid,
            cv=OversampledKFold(self.resampler_),
            refit=False,
        )
        cv_rationale_model.fit(X, Y)

        self.rationale_model = self.base_model(**cv_rationale_model.best_params_)
        self.rationale_model.fit(X, Y)

    def _featurize(self, doc):
        doc_vect = np.asarray([
            token.vector
            for token in doc
            if self._valid(token)
        ])
        rationale_weights = self.rationale_model.predict_proba(doc_vect)[:,1]
        reweighted_doc_vect = np.sum(rationale_weights.reshape(-1, 1) * doc_vect, axis=0)

        if not np.any(np.nonzero(doc_vect)):
            return np.zeros_like(doc.vector)
        else:
            return reweighted_doc_vect / np.linalg.norm(reweighted_doc_vect)
