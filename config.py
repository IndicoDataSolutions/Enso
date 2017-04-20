"""Constants to configure the rest of Enso."""

FEATURIZERS = {
    "IndicoFinance",
    "IndicoStandard",
    "IndicoSentiment",
    "IndicoTopics"
}

DATA = {
    "Classify/TextSpam"
}

EXPERIMENTS = {
    "GridSearchLR",
}

METRICS = {
    "RocAuc"
}

TEST_SETUP = {
    "train_sizes": [25, 50, 100, 250, 500, 1000],
    "n_splits": 5
}
