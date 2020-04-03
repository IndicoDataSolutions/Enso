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
from typing import Any, Tuple

from enso.experiment.tiny_net import RationaleProto, BothProto, BetterLabelBinarizer
import haiku as hk
from jax.experimental import optimizers
import jax.numpy as jnp
import jax


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
            output.append({k: safe_mean([s["confidence"][k] for s in sample]) + 1e-10 for k in classes})
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
            self.NLP = spacy.load("en_vectors_web_lg")

    def _compute_p_rationale(self, docs, rationale_docs):
        word_counts = Counter([token.text.lower() for doc in docs for token in doc])
        rationale_word_counts = Counter([token.text.lower() for doc in rationale_docs for token in doc])

        # smoothing for unseen terms
        base_freq = np.sqrt(1.0 / sum(word_counts.values()))
        self.p_rationale_given_word = defaultdict(lambda: base_freq)

        for word, count in rationale_word_counts.items():
            # there can be slight difference in tokenization -- so we set
            # word counts to be the rationale word count when we encounter this
            self.p_rationale_given_word[word] = np.sqrt(count / word_counts.get(word, count))

        return self.p_rationale_given_word

    def _featurize(self, doc):
        doc_vect = np.mean([token.vector * self.p_rationale_given_word[token.text.lower()] for token in doc], axis=0)
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
        rationale_texts = [rationale["text"] for doc in rationales for rationale in doc]
        docs = np.asarray([self.NLP(str(x), disable=["ner", "tagger", "textcat"]) for x in X])
        rationale_docs = np.asarray([self.NLP(rationale) for rationale in rationale_texts])
        self._compute_p_rationale(docs, rationale_docs)

        doc_vects = np.asarray([self._featurize(doc) for doc in docs])
        resampled_x, resampled_y = self.resample(doc_vects, labels)
        self.model.fit(resampled_x, resampled_y)

    def predict(self, X, **kwargs):
        docs = np.asarray([self.NLP(str(x), disable=["ner", "tagger", "textcat"]) for x in X])
        doc_vects = np.asarray([self._featurize(doc) for doc in docs])
        probas = self.model.predict_proba(doc_vects)
        labels = self.model.classes_

        return pd.DataFrame({label: probas[:, i] for i, label in enumerate(labels)})


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
            self.NLP = spacy.load("en_vectors_web_lg")

    def _valid(self, token):
        return token.has_vector and not token.is_stop and np.any(np.nonzero(token.vector))

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
        rationale_docs = np.asarray([self.NLP(rationale) for rationale in rationale_texts if len(rationale)])
        self._train_rationale_model(docs, rationale_docs)

        doc_vects = np.asarray([self._featurize(doc) for doc in docs])
        resampled_x, resampled_y = self.resample(doc_vects, labels)
        super().fit(resampled_x, resampled_y)

    def predict(self, X, **kwargs):
        docs = np.asarray([self.NLP(str(x), disable=["ner", "tagger", "textcat"]) for x in X])
        doc_vects = np.asarray([self._featurize(doc) for doc in docs])
        probas = self.best_model.predict_proba(doc_vects)
        labels = self.best_model.classes_
        return pd.DataFrame({label: probas[:, i] for i, label in enumerate(labels)})


@Registry.register_experiment(ModeKeys.RATIONALIZED, requirements=[("Featurizer", "PlainTextFeaturizer")])
class DistReweightedGloveClassifierCV(BaseRationaleGridSearch):
    def _train_rationale_model(self, docs, rationale_docs):
        rationale_vecs = [
            doc.vector / np.linalg.norm(doc.vector)
            for doc in rationale_docs
            if doc.has_vector and np.any(np.nonzero(doc.vector))
        ]
        rationale_proto = np.mean(rationale_vecs, axis=0)
        self.normalized_rationale_proto = rationale_proto / np.linalg.norm(rationale_proto)

    def _rationale_weight(self, word):
        cosine_sim = np.dot(word.vector / np.linalg.norm(word.vector), self.normalized_rationale_proto)
        return (1 + cosine_sim) / 2.0

    def _featurize(self, doc):
        doc_vect = np.mean(
            [token.vector * self._rationale_weight(token) for token in doc if self._valid(token)], axis=0
        )

        if not np.any(np.nonzero(doc_vect)):
            return np.zeros_like(doc.vector)
        else:
            return doc_vect / np.linalg.norm(doc_vect)


@Registry.register_experiment(ModeKeys.RATIONALIZED, requirements=[("Featurizer", "PlainTextFeaturizer")])
class RationaleInformedLRCV(BaseRationaleGridSearch):
    def _train_rationale_model(self, docs, rationale_docs):
        rationale_vecs = [
            doc.vector / np.linalg.norm(doc.vector)
            for doc in rationale_docs
            if doc.has_vector and np.any(np.nonzero(doc.vector))
        ]
        rationale_targets = [1] * len(rationale_vecs)
        background_vecs = [
            doc.vector / np.linalg.norm(doc.vector)
            for doc in rationale_docs
            if doc.has_vector and np.any(np.nonzero(doc.vector))
        ]
        background_targets = [0] * len(rationale_vecs)
        X = rationale_vecs + background_vecs
        Y = rationale_targets + background_targets

        cv_rationale_model = GridSearchCV(
            self.base_model(), param_grid=self.param_grid, cv=OversampledKFold(self.resampler_), refit=False,
        )
        cv_rationale_model.fit(X, Y)

        self.rationale_model = self.base_model(**cv_rationale_model.best_params_)
        self.rationale_model.fit(X, Y)

    def _featurize(self, doc):
        doc_vect = np.asarray([token.vector for token in doc if self._valid(token)])
        rationale_weights = self.rationale_model.predict_proba(doc_vect)[:, 1]
        reweighted_doc_vect = np.sum(rationale_weights.reshape(-1, 1) * doc_vect, axis=0)

        if not np.any(np.nonzero(doc_vect)):
            return np.zeros_like(doc.vector)
        else:
            return reweighted_doc_vect / np.linalg.norm(reweighted_doc_vect)


@Registry.register_experiment(ModeKeys.RATIONALIZED, requirements=[("Featurizer", "PlainTextFeaturizer")])
class JaxBase(ClassificationExperiment):

    NLP = None

    def __init__(self, *args, **kwargs):
        super().__init__(auto_resample=False, *args, **kwargs)
        if self.NLP is None:
            self.NLP = spacy.load("en_vectors_web_lg")
        self.target_encoder = BetterLabelBinarizer()
        self.hparams = {"n_epochs": 500, "alpha": 0.5, "l2_coef": 0.01}

    def _token_in_rationales(self, token, rationales):
        for rationale in rationales:
            if (rationale["start"] <= token.idx) and ((token.idx + len(token.text)) <= rationale["end"]):
                return True
        return False

    def train_iter(self, doc_vectors, rationale_targets, targets, batch_size=2, n_epochs=100):
        for _ in range(n_epochs):
            n_total = len(doc_vectors)
            for batch_start in range(0, n_total, batch_size):
                batch_end = min(batch_start + batch_size, n_total)
                bsize = batch_end - batch_start
                doc_vector_slice = doc_vectors[batch_start:batch_end]
                rationale_target_slice = rationale_targets[batch_start:batch_end]
                target_slice = np.asarray(targets[batch_start:batch_end])
                lengths = list(map(len, doc_vector_slice))
                max_length = max(lengths)
                doc_tensor = np.zeros((bsize, max_length, 300))
                rationale_tensor = np.zeros((bsize, max_length, rationale_targets[0].shape[-1]))
                for i, (doc_vector, rationale_target) in enumerate(zip(doc_vector_slice, rationale_target_slice)):
                    doc_tensor[i, : doc_vector.shape[0], :] = doc_vector
                    rationale_tensor[i, : rationale_target.shape[0], :] = rationale_target

                length_mask = np.zeros((bsize, max_length))
                for i in range(bsize):
                    length_mask[i, : lengths[i]] = 1.0

                yield (doc_tensor, rationale_tensor, target_slice, length_mask)

    def predict_iter(self, doc_vectors, batch_size=100):
        n_total = len(doc_vectors)
        for batch_start in range(0, n_total, batch_size):
            batch_end = min(batch_start + batch_size, n_total)
            bsize = batch_end - batch_start
            doc_vector_slice = doc_vectors[batch_start:batch_end]
            lengths = list(map(len, doc_vector_slice))
            max_length = max(lengths)
            doc_tensor = np.zeros((bsize, max_length, 300))
            for i, doc_vector in enumerate(doc_vector_slice):
                doc_tensor[i, : doc_vector.shape[0], :] = doc_vector

            length_mask = np.zeros((bsize, max_length))
            for i in range(bsize):
                length_mask[i, : lengths[i]] = 1.0

            yield (doc_tensor, length_mask)

    def featurize_x(self, X):
        doc_vectors = []
        for text in X:
            doc = self.NLP(text, disable=["ner", "tagger", "textcat"])
            word_vectors = np.zeros((len(doc), 300))
            for i, token in enumerate(doc):
                word_vectors[i, :] = token.vector
            doc_vectors.append(word_vectors)
        return doc_vectors

    def featurize_x_y(self, X, Y):
        doc_vectors = []
        rationale_targets = []
        targets = []
        for text, (rationales, class_target) in zip(X, Y):
            doc = self.NLP(text, disable=["ner", "tagger", "textcat"])
            word_vectors = np.zeros((len(doc), 300))
            rationale_one_hot = np.zeros((len(doc), self.n_classes))
            target_one_hot = self.target_encoder.transform([class_target])[0]
            for i, token in enumerate(doc):
                word_vectors[i, :] = token.vector
                if self._token_in_rationales(token, rationales):
                    rationale_one_hot[i, :] = target_one_hot

            doc_vectors.append(word_vectors)
            rationale_targets.append(rationale_one_hot)
            targets.append(target_one_hot)

        # (batch, seq, hidden) * (batch, seq, one_hot)
        stacked_docs = np.vstack(doc_vectors)
        stacked_targets = np.vstack(rationale_targets)

        rationale_proto = np.einsum("sh,st->ht", stacked_docs, stacked_targets)
        rationale_proto /= np.einsum("st->t", stacked_targets)

        return doc_vectors, rationale_targets, targets, rationale_proto

    def fit(self, X, Y):
        # No rationale means everything is a rationale
        for x, y in zip(X, Y):
            if not y[0]:
                y[0].append({"start": 0, "end": len(x), "label": y[1], "text": x})

        self.target_encoder.fit([y[1] for y in Y])
        self.n_classes = len(self.target_encoder.classes_)

        doc_vectors, rationale_targets, targets, proto = self.featurize_x_y(X, Y)
        train = self.train_iter(doc_vectors, rationale_targets, targets, batch_size=4, n_epochs=1000)

        model_fn = lambda x, length_mask: self.base_model(
            kernel_size=1, n_classes=self.n_classes, rationale_proto=proto
        )(x, length_mask)
        model = hk.transform(model_fn)
        rng = jax.random.PRNGKey(0)
        sample_data_point = next(train)
        params = model.init(rng, sample_data_point[0], sample_data_point[-1])

        opt_init, opt_update, get_params = optimizers.adam(1e-3)
        opt_state = opt_init(params)

        @jax.jit
        def loss(
            params: hk.Params,
            inputs: np.ndarray,
            rationale_targets: np.ndarray,
            targets: np.ndarray,
            length_mask: np.ndarray,
        ):
            batch_size = inputs.shape[0]

            rationale_log_probs, clf_log_probs = model.apply(params, x=inputs, length_mask=length_mask)
            rationale_xent = -jnp.mean(rationale_targets * rationale_log_probs * jnp.expand_dims(length_mask, -1))
            clf_xent = -jnp.mean(targets * clf_log_probs)
            loss = self.hparams["alpha"] * rationale_xent + (1 - self.hparams["alpha"]) * clf_xent
            regularization_loss = self.hparams["l2_coef"] * jax.experimental.optimizers.l2_norm(params)
            return loss + regularization_loss

        @jax.jit
        def update(
            params: hk.Params,
            opt_state: Any,
            inputs: np.ndarray,
            rationale_targets: np.ndarray,
            targets: np.ndarray,
            length_mask: np.ndarray,
            step: int,
        ) -> Tuple[hk.Params, Any]:
            current_loss, gradient = jax.value_and_grad(loss)(
                params, inputs=inputs, rationale_targets=rationale_targets, targets=targets, length_mask=length_mask
            )
            opt_state = opt_update(step, gradient, opt_state)
            new_params = get_params(opt_state)
            return current_loss, new_params, opt_state

        @jax.jit
        def predict(params: hk.Params, inputs: np.ndarray, length_mask: np.ndarray,) -> Tuple[hk.Params, Any]:
            _, clf_log_probs = model.apply(params, x=inputs, length_mask=length_mask)
            return jnp.exp(clf_log_probs)

        self.predict_fn = predict

        step = 0
        for (x_batch, rationale_target_batch, target_batch, length_mask_batch) in train:
            current_loss, params, opt_state = update(
                params, opt_state, x_batch, rationale_target_batch, target_batch, length_mask_batch, step
            )
            step += 1

        self.params = params

    def predict(self, X, **kwargs):
        doc_vectors = self.featurize_x(X)
        probas = []
        data_iter = self.predict_iter(doc_vectors, batch_size=100)
        for batch_vectors, batch_length_mask in data_iter:
            batch_probas = self.predict_fn(params=self.params, inputs=batch_vectors, length_mask=batch_length_mask)
            probas.extend(batch_probas.tolist())
        probas = np.asarray(probas)
        labels = self.target_encoder.classes_
        df = pd.DataFrame({label: probas[:, i] for i, label in enumerate(labels)})
        return df


@Registry.register_experiment(ModeKeys.RATIONALIZED, requirements=[("Featurizer", "PlainTextFeaturizer")])
class Proto(JaxBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model = RationaleProto


@Registry.register_experiment(ModeKeys.RATIONALIZED, requirements=[("Featurizer", "PlainTextFeaturizer")])
class ProtoV2(JaxBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model = BothProto
