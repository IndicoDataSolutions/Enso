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
EXPERIMENT_NAME = "Refactoring"

# Datasets to featurize or run experiments on
DATA = {
    'Classify/AirlineNegativity.medium',
    'Classify/Irony',
    'Classify/Disaster.medium',
    'Classify/Horror.small',
    'Classify/Reddit.10cls.1000',
    'Classify/Reddit.20cls.500',
    'Classify/Reddit.5cls.1000',
    'Classify/ReligiousTexts',
    'Classify/TextSpam',
    'Classify/Disaster',
    'Classify/IMDB.small',
}

# Featurizers to activate
FEATURIZERS = {
    # "SpacyGloveFeaturizer",
    "SpacyCNNFeaturizer",
    # "IndicoStandard"
}

# Experiments to run
EXPERIMENTS = {
    "LogisticRegressionCV",
    "NaiveBayes",
    "RandomForestCV",
    "SupportVectorMachineCV",
}

# Metrics to compute
METRICS = {
    "LogLoss",
    "Accuracy",
    "MacroRocAuc",
}

# Test setup metadata
TEST_SETUP = {
    "train_sizes": [30, 50, 100, 200, 500],
    "n_splits": 10,
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
