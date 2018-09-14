import os
import json

import pandas as pd
import numpy as np
import spacy
from indicoio.custom import Collection
from finetune import Classifier, SequenceLabeler
#from finetune.config import get_default_config, Ranged

from enso.experiment import ClassificationExperiment
from enso.config import RESULTS_DIRECTORY
from enso.registry import Registry, ModeKeys

from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import numpy as np

from enso.utils import labels_to_binary

NLP = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])


def rocauc(result, ground_truth):
    result = pd.DataFrame(result)
    ground_truth = list(ground_truth)
    binary_labels = labels_to_binary(ground_truth)
    binary_labels = np.hstack([binary_labels[column].values.reshape(-1, 1) for column in result.columns])
    predicted_labels = np.hstack([result[column].values.reshape(-1, 1) for column in result.columns])
    return roc_auc_score(binary_labels, predicted_labels, average='macro')

@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class Finetune(ClassificationExperiment):
    """
    LanguageModel finetuning as an alternative to simple models trained on top of pretrained features.
    """

    param_grid = {}

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.model = Classifier(autosave_path=os.path.join(RESULTS_DIRECTORY, '.autosave'),
                                val_size=0.0)

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


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class Finetune2Layers(Finetune):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.config.trainable_layers = [False] * 10 + [True] * 2
        self.model.config.trainable_old_embeddings = False
        self.model.config.trainable_new_embeddings = False
        self.model.config.init_embeddings_from_file = "embeddings.npy"


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class Finetune4Layers(Finetune):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.config.trainable_layers = [False] * 8 + [True] * 4
        self.model.config.trainable_old_embeddings = False
        self.model.config.trainable_new_embeddings = False
        self.model.config.init_embeddings_from_file = "embeddings.npy"


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class Finetune6Layers(Finetune):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.config.trainable_layers = [False] * 6 + [True] * 6
        self.model.config.trainable_old_embeddings = False
        self.model.config.trainable_new_embeddings = False
        self.model.config.init_embeddings_from_file = "embeddings.npy"


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class Finetune8Layers(Finetune):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.config.trainable_layers = [False] * 4 + [True] * 8
        self.model.config.trainable_old_embeddings = False
        self.model.config.trainable_new_embeddings = False
        self.model.config.init_embeddings_from_file = "embeddings.npy"


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class Finetune10Layers(Finetune):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.config.trainable_layers = [False] * 2 + [True] * 10
        self.model.config.trainable_old_embeddings = False
        self.model.config.trainable_new_embeddings = False
        self.model.config.init_embeddings_from_file = "embeddings.npy"


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class Finetune2(Finetune):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.config.trainable_layers = [True] * 2
        self.model.config.trainable_old_embeddings = False
        self.model.config.trainable_new_embeddings = False
        self.model.config.init_embeddings_from_file = "embeddings.npy"

@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class Finetune2Summative(Finetune):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.config.trainable_layers = [True] * 2
        self.model.config.trainable_old_embeddings = False
        self.model.config.trainable_new_embeddings = False
        self.model.config.init_embeddings_from_file = "embeddings.npy"
        self.model.config.reduce_states="SUM"


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class Finetune2Mean(Finetune):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.config.trainable_layers = [True] * 2
        self.model.config.trainable_old_embeddings = False
        self.model.config.trainable_new_embeddings = False
        self.model.config.init_embeddings_from_file = "embeddings.npy"
        self.model.config.reduce_states="Mean"


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class Finetune4(Finetune):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.config.trainable_layers = [True] * 4
        self.model.config.trainable_old_embeddings = False
        self.model.config.trainable_new_embeddings = False
        self.model.config.init_embeddings_from_file = "embeddings.npy"


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class Finetune6(Finetune):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.config.trainable_layers = [True] * 6
        self.model.config.trainable_old_embeddings = False
        self.model.config.trainable_new_embeddings = False
        self.model.config.init_embeddings_from_file = "embeddings.npy"


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class Finetune8(Finetune):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.config.trainable_layers = [True] * 8
        self.model.config.trainable_old_embeddings = False
        self.model.config.trainable_new_embeddings = False
        self.model.config.init_embeddings_from_file = "embeddings.npy"


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class Finetune10(Finetune):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.config.trainable_layers = [True] * 10
        self.model.config.trainable_old_embeddings = False
        self.model.config.trainable_new_embeddings = False
        self.model.config.init_embeddings_from_file = "embeddings.npy"


@Registry.register_experiment(ModeKeys.SEQUENCE, requirements=[("Featurizer", "PlainTextFeaturizer")])
class FinetuneSequenceLabel(ClassificationExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = SequenceLabeler(autosave_path=os.path.join(RESULTS_DIRECTORY, '.autosave'), val_size=0, class_weight='linear')

    def fit(self, X, y):
        for doc, seq in zip(X, y):
            for token in seq:
                token['text'] = doc[token['start']:token['end']]
        corruptX, corruptY = super().resample(X, y)
        self.model.fit(corruptX, corruptY)

    def predict(self, X, **kwargs):
        return self.model.predict(X)


def _convert_to_token_list(annotations, doc_idx=None):
    tokens = []

    for annotation in annotations:
        start_idx = annotation.get('start')
        tokens.extend([
            {
                'start': start_idx + token.idx,
                'end': start_idx + token.idx + len(token.text),
                'text': token.text,
                'label': annotation.get('label'),
                'doc_idx': doc_idx,
                'confidence': annotation.get('confidence')
            }
            for token in NLP(annotation.get('text'))
        ])

    return tokens

def likely_errors(true, predicted, n_corrections=10):
    """
    Return FP, FN, and TP counts
    """

    unique_classes = set([seq['label'] for seqs in true for seq in seqs])

    d = {
        cls_: {
            'false_positives': [],
            'false_negatives': [],
            'correct': []
        }
        for cls_ in unique_classes
    }
    
    labeling_errors = []
    for i, (true_list, pred_list) in enumerate(zip(true, predicted)):
        true_tokens = _convert_to_token_list(true_list, doc_idx=i)
        pred_tokens = _convert_to_token_list(pred_list, doc_idx=i)

        # correct + false negatives
        for true_token in true_tokens:
            for pred_token in pred_tokens:
                if (pred_token['start'] == true_token['start'] and 
                    pred_token['end'] == true_token['end']):

                    if pred_token['label'] == true_token['label']:
                        d[true_token['label']]['correct'].append(true_token)
                    else:
                        d[true_token['label']]['false_negatives'].append(true_token)
                        d[pred_token['label']]['false_positives'].append(pred_token)
                        labeling_errors.append(pred_token)
                    
                    break
            else:
                d[true_token['label']]['false_negatives'].append(true_token)

        # false positives
        for pred_token in pred_tokens:
            for true_token in true_tokens:
                if (pred_token['start'] == true_token['start'] and 
                    pred_token['end'] == true_token['end']):
                    break
            else:
                d[pred_token['label']]['false_positives'].append(pred_token)
                labeling_errors.append(pred_token)
    max_response = lambda token: max(token.get('confidence').values())
    return list(sorted(labeling_errors, key=max_response, reverse=True)[:n_corrections])

    
@Registry.register_experiment(ModeKeys.SEQUENCE, requirements=[("Featurizer", "PlainTextFeaturizer")])
class FinetuneSequenceRelabeled(ClassificationExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = SequenceLabeler(autosave_path=os.path.join(RESULTS_DIRECTORY, '.autosave'), val_size=0, class_weight='linear')

    def fit(self, X, y):
        # corrupt labels
        for doc, seq in zip(X, y):
            for token in seq:
                token['text'] = doc[token['start']:token['end']]
        corruptX, corruptY = super().resample(X, y)
        self.model.fit(corruptX, corruptY)
        n_labels = sum([len(seq) for seq in corruptY])
        predictions = self.model.predict_proba(corruptX)
        corrections = likely_errors(corruptY, predictions, n_corrections=n_labels // 10)
        for correction in corrections:
            for label in y[correction.get('doc_idx')]:
                if label['start'] <= correction['start'] and label['end'] <= correction['end']:
                    corruptY[correction.get('doc_idx')].append(label)
        self.model.fit(corruptX, corruptY)

    def predict(self, X, **kwargs):
        return self.model.predict(X)


@Registry.register_experiment(ModeKeys.SEQUENCE, requirements=[("Featurizer", "PlainTextFeaturizer")])
class FinetuneSequenceCompleteRefit(ClassificationExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = SequenceLabeler(autosave_path=os.path.join(RESULTS_DIRECTORY, '.autosave'), val_size=0, class_weight='linear')

    def fit(self, X, y):
        # corrupt labels
        for doc, seq in zip(X, y):
            for token in seq:
                token['text'] = doc[token['start']:token['end']]
        corruptX, corruptY = super().resample(X, y)
        self.model.fit(corruptX, corruptY)
        n_labels = sum([len(seq) for seq in corruptY])
        predictions = self.model.predict_proba(corruptX)
        corrections = likely_errors(corruptY, predictions, n_corrections=n_labels // 10)
        for correction in corrections:
            for label in y[correction.get('doc_idx')]:
                if label['start'] <= correction['start'] and label['end'] <= correction['end']:
                    corruptY[correction.get('doc_idx')].append(label)
        self.model = SequenceLabeler(autosave_path=os.path.join(RESULTS_DIRECTORY, '.autosave'), val_size=0)
        self.model.fit(corruptX, corruptY)

    def predict(self, X, **kwargs):
        return self.model.predict(X)



@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class Finetune2CV(ClassificationExperiment):
    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.base_conf = get_default_config()
        self.base_conf.update(
            l2_reg=Ranged(0.0, [0.0, 0.001, 0.01, 0.1]),
            trainable_layers=[True] * 2,
            trainable_old_embeddings=False,
            trainable_new_embeddings=False,
            init_embeddings_from_file="embeddings.npy",
            val_size=0.0,
        )
        self.model = None

    def fit(self, X, y):
        """
        :param X: `np.ndarray` of raw text sampled from training data.
        :param y: `np.ndarray` of corresponding targets sampled from training data.
        """
        res = Classifier.finetune_grid_search(
            self.base_conf, [X], y,
            lambda y1, y2: np.mean(np.asarray(y1) == np.asarray(y2)), 0.1)
        self.model = Classifier(res)
        self.model.fit(X, Y=y)

    def predict(self, X, **kwargs):
        """Predict results on test set based on current internal model."""
        preds = self.model.predict_proba(X)
        return pd.DataFrame.from_records(preds)


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class Finetune2CVRoc(ClassificationExperiment):
    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.base_conf = get_default_config()
        self.base_conf.update(
            l2_reg=Ranged(0.0, [0.0, 0.001, 0.01, 0.1]),
            trainable_layers=[True] * 2,
            trainable_old_embeddings=False,
            trainable_new_embeddings=False,
            init_embeddings_from_file="embeddings.npy",
            val_size=0.0,
        )
        self.model = None

    def fit(self, X, y):
        """
        :param X: `np.ndarray` of raw text sampled from training data.
        :param y: `np.ndarray` of corresponding targets sampled from training data.
        """
        res = Classifier.finetune_grid_search(self.base_conf, [X], y, rocauc, 0.1, probs=True)
        self.model = Classifier(res)
        self.model.fit(X, Y=y)

    def predict(self, X, **kwargs):
        """Predict results on test set based on current internal model."""
        preds = self.model.predict_proba(X)
        return pd.DataFrame.from_records(preds)


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class FinetuneLast2CV(ClassificationExperiment):
    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.base_conf = get_default_config()
        self.base_conf.update(
            l2_reg=Ranged(0.0, [0.0, 0.001, 0.01, 0.1]),
            trainable_layers=[False] * 10 + [True] * 2,
            trainable_old_embeddings=False,
            trainable_new_embeddings=False,
            init_embeddings_from_file="embeddings.npy",
            val_size=0.0,
        )
        self.model = None

    def fit(self, X, y):
        """
        :param X: `np.ndarray` of raw text sampled from training data.
        :param y: `np.ndarray` of corresponding targets sampled from training data.
        """
        res = Classifier.finetune_grid_search(
            self.base_conf, [X], y,
            lambda y1, y2: np.mean(np.asarray(y1) == np.asarray(y2)), 0.1)
        self.model = Classifier(res)
        self.model.fit(X, Y=y)

    def predict(self, X, **kwargs):
        """Predict results on test set based on current internal model."""
        preds = self.model.predict_proba(X)
        return pd.DataFrame.from_records(preds)

    @Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
    class Finetune8CV(ClassificationExperiment):
        def __init__(self, *args, **kwargs):
            """Initialize internal classifier."""
            super().__init__(*args, **kwargs)
            self.base_conf = get_default_config()
            self.base_conf.update(
                l2_reg=Ranged(0.0, [0.0, 0.001, 0.01, 0.1]),
                trainable_layers=[True] * 8,
                trainable_old_embeddings=False,
                trainable_new_embeddings=False,
                init_embeddings_from_file="embeddings.npy",
                val_size=0.0,
            )
            self.model = None

        def fit(self, X, y):
            """
            :param X: `np.ndarray` of raw text sampled from training data.
            :param y: `np.ndarray` of corresponding targets sampled from training data.
            """
            res = Classifier.finetune_grid_search(
                self.base_conf, [X], y,
                lambda y1, y2: np.mean(np.asarray(y1) == np.asarray(y2)), 0.1)
            self.model = Classifier(res)
            self.model.fit(X, Y=y)

        def predict(self, X, **kwargs):
            """Predict results on test set based on current internal model."""
            preds = self.model.predict_proba(X)
            return pd.DataFrame.from_records(preds)

    @Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
    class FinetuneLast8CV(ClassificationExperiment):
        def __init__(self, *args, **kwargs):
            """Initialize internal classifier."""
            super().__init__(*args, **kwargs)
            self.base_conf = get_default_config()
            self.base_conf.update(
                l2_reg=Ranged(0.0, [0.0, 0.001, 0.01, 0.1]),
                trainable_layers=[False] * 4 + [True] * 8,
                trainable_old_embeddings=False,
                trainable_new_embeddings=False,
                init_embeddings_from_file="embeddings.npy",
                val_size=0.0,
            )
            self.model = None

        def fit(self, X, y):
            """
            :param X: `np.ndarray` of raw text sampled from training data.
            :param y: `np.ndarray` of corresponding targets sampled from training data.
            """
            res = Classifier.finetune_grid_search(
                self.base_conf, [X], y,
                lambda y1, y2: np.mean(np.asarray(y1) == np.asarray(y2)), 0.1)
            self.model = Classifier(res)
            self.model.fit(X, Y=y)

        def predict(self, X, **kwargs):
            """Predict results on test set based on current internal model."""
            preds = self.model.predict_proba(X)
            return pd.DataFrame.from_records(preds)



    @Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
    class Finetune4LayersCV(ClassificationExperiment):
        def __init__(self, *args, **kwargs):
            """Initialize internal classifier."""
            super().__init__(*args, **kwargs)
            self.base_conf = get_default_config()
            self.base_conf.update(
                l2_reg=Ranged(0.0, [0.0, 0.001, 0.01, 0.1]),
                trainable_layers=[False] * 8 + [True] * 4,
                trainable_old_embeddings=False,
                trainable_new_embeddings=False,
                init_embeddings_from_file="embeddings.npy",
                val_size=0.0,
            )
            self.model = None

        def fit(self, X, y):
            """
            :param X: `np.ndarray` of raw text sampled from training data.
            :param y: `np.ndarray` of corresponding targets sampled from training data.
            """
            res = Classifier.finetune_grid_search(
                self.base_conf, [X], y,
                lambda y1, y2: np.mean(np.asarray(y1) == np.asarray(y2)), 0.1)
            self.model = Classifier(res)
            self.model.fit(X, Y=y)

        def predict(self, X, **kwargs):
            """Predict results on test set based on current internal model."""
            preds = self.model.predict_proba(X)
            return pd.DataFrame.from_records(preds)

    @Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
    class FinetuneCVNumLayers(ClassificationExperiment):
        def __init__(self, *args, **kwargs):
            """Initialize internal classifier."""
            super().__init__(*args, **kwargs)
            self.base_conf = get_default_config()
            self.base_conf.update(
                l2_reg=0.1,
                trainable_layers=Ranged([True] * 12, [[True] * i for i in range(0, 13, 2)] + [[False] * (12 - i) + [True] * i for i in range(0, 13, 2)]),
                trainable_old_embeddings=False,
                trainable_new_embeddings=False,
                init_embeddings_from_file="embeddings.npy",
                val_size=0.0,
            )
            self.model = None

        def fit(self, X, y):
            """
            :param X: `np.ndarray` of raw text sampled from training data.
            :param y: `np.ndarray` of corresponding targets sampled from training data.
            """
            res = Classifier.finetune_grid_search(
                self.base_conf, [X], y,
                lambda y1, y2: np.mean(np.asarray(y1) == np.asarray(y2)), 0.1)
            with open("CONFIGS", "at") as fp:
                fp.write(str(res.trainable_layers) + "\n")
            self.model = Classifier(res)
            self.model.fit(X, Y=y)

        def predict(self, X, **kwargs):
            """Predict results on test set based on current internal model."""
            preds = self.model.predict_proba(X)
            return pd.DataFrame.from_records(preds)


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

    def __del__(self):
        self.model.clear()
1
