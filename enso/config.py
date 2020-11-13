import multiprocessing
import functools

import indicoio

from enso.mode import ModeKeys

"""Constants to configure the rest of Enso."""

# Directory for storing data
DATA_DIRECTORY = "Data"

# Directory for storing results
RESULTS_DIRECTORY = "Results"

# Directory for storing features
FEATURES_DIRECTORY = "Features"

# Directory for storing experiment results
EXPERIMENT_NAME = "BERT-Showdown"

# Name of the csv used to store results
RESULTS_CSV_NAME = "Results.csv"

# Datasets to featurize or run experiments on
DATA = {
    # "SequenceLabeling/Reuters-128",
    # "SequenceLabeling/bonds",
    # "SequenceLabeling/bonds_new",
    "SequenceLabeling/cord",
    "SequenceLabeling/invoices",
    # "SequenceLabeling/correspondence",
    # "SequenceLabeling/d_invoices",
    # "SequenceLabeling/C_data",
    # "SequenceLabeling/H_data",
    # "SequenceLabeling/wiki",
}

# Featurizers to activate
FEATURIZERS = {
    "PlainTextFeaturizer",
}

# Experiments to run
EXPERIMENTS = {
    # "FinetuneRoberta",
    "FinetuneAlbert"
}

# Metrics to compute
METRICS = {
    "MicroCharF1",
    "MicroCharRecall",
    "MicroCharPrecision",
    "MacroCharF1",
    "MacroCharRecall",
    "MacroCharPrecision",
}

# Test setup metadata
TEST_SETUP = {
    "train_sizes": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "n_splits": 3,
    "samplers": ["RandomSequence"],
    "sampling_size": 0.2,
    "resamplers": ["NoResampler"],
}

# Visualizations to display
VISUALIZATIONS = {"FacetGridVisualizer"}

# kwargs to pass directly into visualizations
VISUALIZATION_OPTIONS = {
    "display": True,
    "save": True,
    "FacetGridVisualizer": {
        "x_tile": "Metric",
        "y_tile": "Dataset",
        "x_axis": "TrainSize",
        "y_axis": "Result",
        "lines": ["Experiment", "Featurizer", "Sampler", "Resampler"],
        "category": "merge",
        "cv": "mean",
        "filename": "TestResult",
    },
}

MODE = ModeKeys.SEQUENCE

N_GPUS = 1
N_CORES = 1  # multiprocessing.cpu_count()

FIX_REQUIREMENTS = True

GOLD_FRAC = 0.05
CORRUPTION_FRAC = 0.4

indicoio.config.api_key = ""

# If we have no experiment hyperparameters we hope to modify:
EXPERIMENT_PARAMS = {}
