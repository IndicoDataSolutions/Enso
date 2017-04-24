"""Constants to configure the rest of Enso."""
# Featurizers to activate
FEATURIZERS = {
    "IndicoFinance",
    "IndicoStandard",
    "IndicoSentiment",
    "IndicoTopics"
}

# Datasets to featurize or run experiments on
DATA = {
    "Classify/TextSpam"
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
    "train_sizes": [25, 50, 100, 250, 500, 1000],
    "n_splits": 5
}

# Visualizations to display
VISUALIZATIONS = {
    'FacetGridVisualization'
}

# kwargs to pass directly into visualizations
VISUALIZATION_OPTIONS = {
    'display': True,
    'save': True,
    'FacetGridVisualization': {
        'x_tile': 'Dataset',
        'y_tile': 'Experiment',
        'x_axis': 'TrainSize',
        'y_axis': 'Result',
        'lines': 'Featurizer',
        'category': 'merge',
        'cv': 'mean'
    }
}
