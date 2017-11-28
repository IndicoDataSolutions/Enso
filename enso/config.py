import indicoio

"""
Configuring indicoio settings to point to local indico API instance
"""
indicoio.config.api_key = "cac939f48044ddf637a974b4baa290ca"
indicoio.config.host = "172.17.0.1:8000"
indicoio.config.url_protocol = "http"

"""Constants to configure the rest of Enso."""
# Directory for storing results
RESULTS_DIRECTORY = "Results"

# Directory for storing features
FEATURES_DIRECTORY = "Features"

# Datasets to featurize or run experiments on
DATA = {
    # "Classify/TextSpam.small",
    # "Classify/IMDB.small",
    # "Classify/HotelReviews.small",
    # "Classify/AirlineNegativity.small",
    # "Classify/Irony",
    "Classify/ReligiousTexts"
}

# Featurizers to activate
FEATURIZERS = {
    "IndicoFinance",
    "IndicoStandard",
    "IndicoSentiment",
    "IndicoTopics",
    "TransformerFeaturizer"
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
    "train_sizes": [25, 50, 100, 250, 500],
    "n_splits": 5
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
        'y_tile': 'Experiment',
        'x_axis': 'TrainSize',
        'y_axis': 'Result',
        'lines': 'Featurizer',
        'category': 'merge',
        'cv': 'mean'
    }
}

