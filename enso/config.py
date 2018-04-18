import indicoio

"""Constants to configure the rest of Enso."""
# Directory for storing results
RESULTS_DIRECTORY = "Results"

# Directory for storing features
FEATURES_DIRECTORY = "Features"

EXPERIMENT_NAME = "SklearnComparison"

# Datasets to featurize or run experiments on
DATA = {
    # 'Classify/AirlineComplaints',
    'Classify/AirlineNegativity.medium',
    'Classify/Irony',
    # 'Classify/Disaster.medium',
    # 'Classify/Horror.small',

    'Classify/Reddit.10cls.1000',
    'Classify/Reddit.20cls.500',
    'Classify/Reddit.5cls.1000',

    # 'Classify/ReligiousTexts',
    # 'Classify/ShortAnswer',
    'Classify/TextSpam',
    'Classify/Disaster',
    # 'Classify/Economy',
    # 'Classify/BIA',
    'Classify/IMDB.small',
    # 'Classify/HotelReviews.small',
}

# Featurizers to activate
FEATURIZERS = {
    "IndicoTransformer",
    # "IndicoStandard",
    # "IndicoSentiment",
    # "IndicoFastText",
    # "IndicoFinance",
    # "IndicoTopics",
    # "ElmoFeaturizer",
    # "IndicoTransformerSequence",
    # "IndicoStandardSequence",
}

# Experiments to run
EXPERIMENTS = {
    # "LR",
    # "NaiveBayes",
    # "RandomForest",
    "RBFSVM",
    # "ReduceMaxClassifier",
    # "NormedDotAttnClassifier",
    # "ReduceMeanClassifier",
    # "FancyL2DotAttnClassifier",
    # "StrongRegDotAttnClassifier",
    # "MulticlassDotAttnClassifier",
    # "MultiheadAttnClassifier",
    # "MultiheadAttnV2Classifier",
    # "RegularizedMLPAttnClassifier",
    # "SummaryStatsClassifier"
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
    "resamplings": ['RandomOverSampler']
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
        'lines': ['Experiment', 'Featurizer', 'Hyperparams'],
        'category': 'merge',
        'cv': 'mean',
        'filename': 'TestResult'
    }
}


N_GPUS = 3
N_CORES = 4
