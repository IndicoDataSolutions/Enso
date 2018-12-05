import indicoio
from enso.mode import ModeKeys
import multiprocessing

"""Constants to configure the rest of Enso."""

# Directory for storing data
DATA_DIRECTORY = "Data"

# Directory for storing results
RESULTS_DIRECTORY = "Results"

# Directory for storing features
FEATURES_DIRECTORY = "Features"

# Directory for storing experiment results
EXPERIMENT_NAME = "BERT"

# Datasets to featurize or run experiments on
DATA = {
    # Classification
    'Classify/AirlineSentiment',
    'Classify/MovieReviews',
    'Classify/MPQA',
    'Classify/PoliticalTweetSubjectivity',

    # Seqence

    # 'SequenceLabeling/Reuters-128',
    # 'SequenceLabeling/brown_all',
    # 'SequenceLabeling/brown_nouns',
    # 'SequenceLabeling/brown_verbs',
    # 'SequenceLabeling/brown_pronouns',
    # 'SequenceLabeling/brown_adverbs',
}

# Featurizers to activate
FEATURIZERS = {
    "BERTFeaturizer",
    # "PlainTextFeaturizer",
    # "IndicoStandard",
    # "SpacyGloveFeaturizer",
    # "SpacyCNNFeaturizer",
}

# Experiments to run
EXPERIMENTS = {
    # "FinetuneSequenceLabel",
    # "IndicoSequenceLabel"
    # "Finetune",
    "BERT"
    # "SpacyGlove"
    # "LogisticRegressionCV"
}

# Metrics to compute
METRICS = {
    # "OverlapAccuracy",
    # "OverlapPrecision",
    # "OverlapRecall",
    "Accuracy",
    "MacroRocAuc",
}

# Test setup metadata
TEST_SETUP = {
    "train_sizes": range(100, 500, 50),
    "n_splits": 5,
    # "samplers": ['RandomSequence', 'NoSampler'],
    "samplers": ['Random'],
    "sampling_size": .3,
    # "resamplers": ["SequenceOverSampler", 'NoResampler']
    "resamplers": ["RandomOverSampler"]
}

# Visualizations to display
VISUALIZATIONS = {
    'FacetGridVisualizer'
}

# kwargs to pass directly into visualizations
VISUALIZATION_OPTIONS = {
    'display': True,
    'save': True,
    'FacetGridVisualizer': {
        'x_tile': 'Metric',
        'y_tile': 'Dataset',
        'x_axis': 'TrainSize',
        'y_axis': 'Result',
        'lines': ['Experiment', 'Featurizer', "Sampler", "Resampler"],
        'category': 'merge',
        'cv': 'mean',
        'filename': 'TestResult'
    }
}

MODE = ModeKeys.CLASSIFY

N_GPUS = 1
N_CORES = 1  # multiprocessing.cpu_count()

FIX_REQUIREMENTS = True

indicoio.config.api_key = ""
