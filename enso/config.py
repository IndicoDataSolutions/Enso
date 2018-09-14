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
EXPERIMENT_NAME = "CorruptedSequence"

# Datasets to featurize or run experiments on
DATA = {
    # Classification
    # 'Classify/AirlineSentiment',
    # 'Classify/MovieReviews',
    # 'Classify/MPQA',
    # 'Classify/PoliticalTweetSubjectivity',

    # Seqence

    # 'SequenceLabeling/Reuters-128',
    # 'SequenceLabeling/brown_all',
    'SequenceLabeling/brown_nouns.small',
    'SequenceLabeling/brown_verbs.small',
    'SequenceLabeling/brown_pronouns.small',
    'SequenceLabeling/brown_adverbs.small',
}

# Featurizers to activate
FEATURIZERS = {
    "PlainTextFeaturizer",
    # "IndicoStandard",
    # "SpacyGloveFeaturizer",
    # "SpacyCNNFeaturizer",
}

# Experiments to run
EXPERIMENTS = {
    "FinetuneSequenceRelabeled",
    # "FinetuneSequenceLabel"
    # "IndicoSequenceLabel"
    # "Finetune",
    # "SpacyGlove"
    # "LogisticRegressionCV"
}

# Metrics to compute
METRICS = {
    "OverlapAccuracy",
    "OverlapPrecision",
    "OverlapRecall",
    # "Accuracy",
    # "MacroRocAuc",
}

# Test setup metadata
TEST_SETUP = {
    "train_sizes": range(50, 500, 50),
    "n_splits": 2,
    "samplers": ['RandomSequence'],
    "sampling_size": .3,
    "resamplers": ["SequenceCorruptiveResampler"]
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

CORRUPTION_FRAC = 0.3
GOLD_FRAC = 0.1

MODE = ModeKeys.SEQUENCE

N_GPUS = 1
N_CORES = 1  # multiprocessing.cpu_count()

FIX_REQUIREMENTS = True

indicoio.config.api_key = ""
