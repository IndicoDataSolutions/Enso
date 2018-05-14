import indicoio
import multiprocessing

"""Constants to configure the rest of Enso."""

# Directory for storing data
DATA_DIRECTORY = "Data"

# Directory for storing results
RESULTS_DIRECTORY = "Results"

# Directory for storing features
FEATURES_DIRECTORY = "Features"

# Directory for storing experiment results
EXPERIMENT_NAME = "Demo"

# Datasets to featurize or run experiments on
DATA = {
    'Classify/AirlineSentiment',
    'Classify/MovieReviews',
    'Classify/MPQA',
    'Classify/PoliticalTweetSubjectivity',
}

# Featurizers to activate
FEATURIZERS = {
    "SpacyGloveFeaturizer",
    "SpacyCNNFeaturizer"
}

# Experiments to run
EXPERIMENTS = {
    "LogisticRegressionCV",
}

# Metrics to compute
METRICS = {
    "Accuracy",
    "MacroRocAuc",
}

# Test setup metadata
TEST_SETUP = {
    "train_sizes": range(50, 550, 50),
    "n_splits": 25,
    "samplers": ['Random'],
    "sampling_size": .3,
    "resamplers": ['RandomOverSampler']
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
        'lines': ['Experiment', 'Featurizer'],
        'category': 'merge',
        'cv': 'mean',
        'filename': 'TestResult'
    }
}


N_GPUS = 3
N_CORES = 1 # multiprocessing.cpu_count()
