import pandas as pd
from sklearn.model_selection import GridSearchCV

from enso.experiment import ClassificationExperiment
from sklearn.linear_model import LogisticRegression
from enso.utils import OversampledKFold
from enso.registry import Registry, ModeKeys
from finetune import Classifier, SequenceLabeler

class RationalizedGridSearch(ClassificationExperiment):
    """
    Base class for classification models that select hyperparameters via cross validation.
    Assumes the `base_model` property set on child classes inherits from
    `sklearn.base.BaseEstimator` and implements `predict_proba` and `score`.

    :cvar base_model: Class name of base model, must be set by child classes.
    :cvar param_grid: Dictionary that maps from class paramaters to an array of values to search over.  Must be set by child classes.
    """

    param_grid = {}
    base_model = None

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(auto_resample=False, *args, **kwargs)
        self.best_model = None

    def fit(self, X, y):
        """
        Runs grid search over `self.param_grid` on `self.base_model` to optimize hyper-parameters using
        KFolds cross-validation, then retrains using the selected parameters on the full training set.

        :param X: `np.ndarray` of input features sampled from training data.
        :param y: `np.ndarray` of corresponding targets sampled from training data.
        """
        classifier = GridSearchCV(
            self.base_model(),
            param_grid=self.param_grid,
            cv=OversampledKFold(self.resampler_),
            refit=False,
        )
        classifier.fit(X, [yi[1] for yi in y])

        self.best_model = self.base_model(**classifier.best_params_)
        self.best_model.fit(*self.resample(X, [yi[1] for yi in y]))

    def predict(self, X, **kwargs):
        """Predict results on test set based on current internal model."""
        labels = self.best_model.classes_
        probabilities = self.best_model.predict_proba(X)
        return pd.DataFrame(
            {label: probabilities[:, i] for i, label in enumerate(labels)}
        )


@Registry.register_experiment(ModeKeys.RATIONALIZED, requirements=[("Featurizer", "not PlainTextFeaturizer")])
class LogisticRegressionCVRationalized(RationalizedGridSearch):
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
class FinetuneClfBaselineRationalized(ClassificationExperiment):
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
        print(classes)
        for sample in preds:
            overall_confidence = {k: safe_mean([s["confidence"][k] for s in sample]) + 1e-10 for k in classes}
            norm_factor = sum(overall_confidence.values())
            overall_conficence = {k: v / norm_factor for k, v in overall_confidence.items()}
            output.append(overall_confidence)
        return pd.DataFrame.from_records(output)

    def cleanup(self):
        del self.model

                                                                                                
