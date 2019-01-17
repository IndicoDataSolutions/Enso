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
EXPERIMENT_NAME = "Benchmark"

# Datasets to featurize or run experiments on
DATA = {
    'Classify/AirlineNegativity',
    'Classify/AirlineSentiment',
    'Classify/BrandEmotion',
    'Classify/BrandEmotionCause',
    'Classify/ChemicalDiseaseCauses',
    'Classify/CorporateMessaging',
    'Classify/CustomerReviews',
    'Classify/DetailedEmotion',
    'Classify/DrugReviewType',
    'Classify/DrugReviewIntent',
    'Classify/Economy',
    'Classify/Emotion',
    'Classify/GlobalWarming',
    'Classify/MovieReviews',
    'Classify/MPQA',
    'Classify/NewYearsResolutions',
    'Classify/PoliticalTweetBias',
    'Classify/PoliticalTweetClassification',
    'Classify/PoliticalTweetAlignment',
    'Classify/PoliticalTweetSubjectivity',
    'Classify/PoliticalTweetTarget',
    'Classify/SocialMediaDisasters',
    'Classify/SST-binary',
    'Classify/Subjectivity'
}

# Featurizers to activate
FEATURIZERS = {
    "SpacyGloveFeaturizer",
    # "SpacyCNNFeaturizer",
}

# Experiments to run
EXPERIMENTS = {
    "LogisticRegressionCV",
    # "NaiveBayes",
    # "RandomForestCV",
    # "SupportVectorMachineCV",
}

# Metrics to compute
METRICS = {
    "LogLoss",
    "Accuracy",
    "MacroRocAuc",
}

# Test setup metadata
TEST_SETUP = {
    "train_sizes": list(range(500, 550, 50)),
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
