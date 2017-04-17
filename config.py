"""Constants to configure the rest of Enso."""

FEATURIZERS = {
    "IndicoTopics",
    "IndicoSentiment",
    "IndicoStandard",
    "IndicoFinance",
}

DATA = {
    "Classify/TextSpam"
}

EXPERIMENTS = {
    "BasicLogisticRegression",
    "BasicNaiveBayes",
}

METRICS = {
    "Accuracy"
}

TEST_SETUP = {
    "train_sizes": [25, 50, 100, 250, 500, 1000],
    "stratified": True,
    "n_splits": 5
}
