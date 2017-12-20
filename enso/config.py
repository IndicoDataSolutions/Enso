import indicoio

"""Constants to configure the rest of Enso."""
# Directory for storing results
RESULTS_DIRECTORY = "Results"

# Directory for storing features
FEATURES_DIRECTORY = "Features"

EXPERIMENT_NAME = "benchmark-v1"

# Datasets to featurize or run experiments on
DATA = {
    #'Classify/AirlineComplaints',
    'Classify/Disaster',
    'Classify/Irony',
    'Classify/IMDB.small',
    'Classify/Economy',
    #'Classify/Emotion',
    # 'Classify/Horror',
    # 'Classify/HotelReviews.small',
    # 'Classify/IdiomEmotion',
    # 'Classify/Reddit.10cls.1000',
    # 'Classify/Reddit.20cls.500',
    # 'Classify/Reddit.5cls.1000',
    # 'Classify/ReligiousTexts',
    #'Classify/ShortAnswer',
    #'Classify/TextSpam'
    # 'Classify/AirlineNegativity.small',
}

# Featurizers to activate
FEATURIZERS = {
    "IndicoStandard",
    #"IndicoFastText",
    # "IndicoTransformer",
    # "IndicoFinance",
    # "IndicoTopics",
    # "IndicoSentiment",
}

# Experiments to run
EXPERIMENTS = {
    "GridSearchLR",
}

# Metrics to compute
METRICS = {
    "RocAuc"
}

# Test setup metadata
TEST_SETUP = {
    "train_sizes": [10, 30, 50, 100, 200],
    "n_splits": 10,
    "samplers": ['Random', 'Orthogonal'],
    "sampling_size": .3,
    "resamplings": ['none', 'SMOTE', 'ADASYN']
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
        'x_tile': 'Dataset',
        'y_tile': 'Sampler',
        'x_axis': 'TrainSize',
        'y_axis': 'Result',
        'lines': 'Resampling',
        'category': 'merge',
        'cv': 'mean'
    }
}


N_CORES = 8
