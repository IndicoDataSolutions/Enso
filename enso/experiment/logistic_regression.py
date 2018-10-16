"""Module for any LR-style experiment."""
from collections import Counter

import pandas as pd
import numpy as np

import spacy

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import RadiusNeighborsRegressor
import tqdm


from enso.experiment.grid_search import GridSearch

from enso.registry import Registry, ModeKeys


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "not PlainTextFeaturizer")])
class LogisticRegressionCV(GridSearch):
    """Implementation of a grid-search optimized Logistic Regression model."""

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.base_model = LogisticRegression
        self.param_grid = {
            'penalty': ['l2'],
            'C': [0.1, 1.0, 10., 100., 1000.],
        }


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class ReweightingLR(GridSearch):
    param_grid = {}
    base_model = None

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.base_model = LogisticRegression
        self.param_grid = {
            'penalty': ['l1'],
            'C': [0.01, 0.1, 1.0, 10., 100., 1000.],
        }
        self.rwe = ReweightingEmbedder()

    def fit(self, X, y):

        self.rwe.fit(X, y)
        X_p = self.rwe.predict(X)

        classifier = GridSearchCV(
            self.base_model(),
            param_grid=self.param_grid
        )
        classifier.fit(X_p, y)

        self.best_model = self.base_model(**classifier.best_params_)
        self.best_model.fit(X_p, y)

    def predict(self, X, **kwargs):
        """Predict results on test set based on current internal model."""
        X_p = self.rwe.predict(X)
        labels = self.best_model.classes_
        probabilities = self.best_model.predict_proba(X_p)
        return pd.DataFrame({label: probabilities[:, i] for i, label in enumerate(labels)})




class ReweightingEmbedder:
    def __init__(self, lemma=True, spacy_vector_model="en_vectors_web_lg", spacy_lemma_model="en_core_web_lg"):
        self.vector_nlp = spacy.load(spacy_vector_model, disable=['parser', 'tagger', 'ner', 'textcat'])

        if lemma:
            self.nlp = spacy.load(spacy_lemma_model, disable=['parser', 'ner', 'textcat'])
        else:
            self.nlp = self.vector_nlp

        self.embed_to_weight = None
        self.weight_cache = None
        self.lemma = lemma

    def token(self, spacy_obj):
        if self.lemma:
            return spacy_obj.lemma_
        return spacy_obj.text

    def fit(self, texts, labels, denoising_threshold=0, neighbourhood_mul=0.01, perc_of_mean=0.5):

        if len(set(labels)) < 2:
            self.predict = lambda doc: [x.vector for x in self.vector_nlp.pipe(doc)]
            print("Only one class provided, standard vectors will be used")
            return

        self.weight_cache = dict()
        docs = [[self.token(t) for t in doc] for doc in self.nlp.pipe(texts)]
        bow = dict([(l, Counter()) for l in labels])
        words_per_label = dict([(l, 0) for l in labels])
        global_counts = Counter()

        calibration_dist = self.vector_nlp.vocab["he"].similarity(self.vector_nlp.vocab["he"]) * neighbourhood_mul
        print("Radius used is: ", calibration_dist)

        for doc, label in zip(docs, labels):
            global_counts.update(doc)

        global_counts_2 = dict()
        for word, value in global_counts.items():
            if value > denoising_threshold:
                global_counts_2[word] = value

        del global_counts
        global_counts = global_counts_2

        for doc, label in zip(docs, labels):
            doc = [w for w in doc if w in global_counts]
            words_per_label[label] += len(doc)
            bow[label].update(doc)

        importances = dict()
        for label in bow:
            per_class_word_importance = dict()
            for k in bow[label].keys():
                word_count = bow[label][k]
                word_class_frequency = word_count / words_per_label[label]
                word_class_importance = word_class_frequency
                per_class_word_importance[k] = word_class_importance
            importances[label] = per_class_word_importance

        self.weighting = dict()
        eps = 1 / sum(words_per_label.values())
        for word in global_counts:
            freqs = sorted((importances[l][word] if word in importances[l] else eps for l in importances), reverse=True)
            print(word, freqs)
            self.weighting[word] = (freqs[0]) / (np.float(freqs[1]))

        vecs = [self.vector_nlp.vocab[word].vector for word in tqdm.tqdm(self.weighting)]
        self.embed_to_weight = RadiusNeighborsRegressor(radius=calibration_dist, metric="cosine")
        self.embed_to_weight.fit(vecs, list(self.weighting.values()))
        self.mean_weighting = np.mean(list(self.weighting.values())) * perc_of_mean

    def predict(self, docs, normalise_by_length=True, lemma_vecs=True):
        predictions = []
        for document in tqdm.tqdm(self.nlp.pipe(docs)):
            vector = self.vector_nlp.vocab["中"].vector  # unk...
            for w in document:
                token = self.token(w)
                if token in self.weight_cache:
                    weight = self.weight_cache[token]
                else:
                    weight = self.embed_to_weight.predict([self.vector_nlp.vocab[token].vector])[0]
                    if not np.isfinite(weight) or w.is_oov:
                        weight = self.mean_weighting
                    self.weight_cache[token] = weight

                word = self.vector_nlp.vocab[
                    w.text if (not lemma_vecs or self.token(w) not in self.vector_nlp.vocab) else self.token(w)]
                vector += weight * word.vector
            if normalise_by_length:
                vector /= len(document)
            predictions.append(vector)
        return predictions
