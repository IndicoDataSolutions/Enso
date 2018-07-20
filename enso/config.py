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
EXPERIMENT_NAME = "Finetune-0.2.0"

# Datasets to featurize or run experiments on
DATA = {
    "Classify/AirlineComplaints",
    "Classify/AirlineNegativity",
    "Classify/AirlineSentiment",
    "Classify/BrandEmotion",
    "Classify/BrandEmotionCause",
    "Classify/ChemicalDiseaseCauses",
    "Classify/CorporateMessaging",
    "Classify/CustomerReviews",
    "Classify/DetailedEmotion",
    "Classify/Disaster",
    "Classify/DrugReviewIntent",
    "Classify/DrugReviewType",
    "Classify/Economy",
    "Classify/Emotion",
    "Classify/GlobalWarming",
    "Classify/Horror",
    "Classify/HotelReviews",
    "Classify/IMDB",
    "Classify/Irony",
    "Classify/MPQA",
    "Classify/MovieReviews",
    "Classify/NewYearsResolutions",
    "Classify/PoliticalTweetAlignment",
    "Classify/PoliticalTweetBias",
    "Classify/PoliticalTweetClassification",
    "Classify/PoliticalTweetSubjectivity",
    "Classify/PoliticalTweetTarget",
    "Classify/Reddit.10cls.1000",
    "Classify/Reddit.20cls.500",
    "Classify/Reddit.50cls.200",
    "Classify/Reddit.5cls.1000",
    "Classify/ReligiousTexts",
    "Classify/ShortAnswer",
    "Classify/SocialMediaDisasters",
    "Classify/Subjectivity",
    "Classify/TextSpam",
}

# Featurizers to activate
FEATURIZERS = {
    # "SpacyGloveFeaturizer",
    # "SpacyCNNFeaturizer"
    "PlainTextFeaturizer"
}

# Experiments to run
EXPERIMENTS = {
    "Finetune",
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
    "resamplers": ["None"]
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
