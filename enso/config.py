import indicoio
from enso.mode import ModeKeys
import multiprocessing

"""Constants to configure the rest of Enso."""

# Directory for storing data
DATA_DIRECTORY = "Data"

# Directory for storing results
RESULTS_DIRECTORY = "Results"

# Directory for storing features
FEATURES_DIRECTORY = "Features"

# Directory for storing experiment results
EXPERIMENT_NAME = "RationalizedTest"

# Name of the csv used to store results
RESULTS_CSV_NAME = "Results.csv"

# Datasets to featurize or run experiments on
DATA = {
#    "Classify/AirlineComplaints",
    # "Classify/AirlineNegativity",
    # "Classify/AirlineSentiment",
    # "Classify/BrandEmotion",
    # "Classify/BrandEmotionCause",
    # "Classify/ChemicalDiseaseCauses",
    # "Classify/CorporateMessaging",
    # "Classify/CustomerReviews",
    # "Classify/DetailedEmotion",
    # "Classify/Disaster",
    # "Classify/DrugReviewIntent",
    # "Classify/DrugReviewType",
    # "Classify/Economy",
    # "Classify/Emotion",
    # "Classify/GlobalWarming",
    # "Classify/Horror",
    # "Classify/HotelReviews",
    # "Classify/IMDB",
    # "Classify/Irony",
    # "Classify/MPQA",
    # "Classify/MovieReviews",
    # "Classify/NewYearsResolutions",
    # "Classify/PoliticalTweetAlignment",
    # "Classify/PoliticalTweetBias",
    # "Classify/PoliticalTweetClassification",
    # "Classify/PoliticalTweetSubjectivity",
    # "Classify/PoliticalTweetTarget",
    # "Classify/ReligiousTexts",
    # "Classify/ShortAnswer",
    # "Classify/SocialMediaDisasters",
    # "Classify/Subjectivity",
    # "Classify/TextSpam",
    # "Classify/SST-binary"
    # Seqence
    # 'SequenceLabeling/Reuters-128',
    # 'SequenceLabeling/brown_all',
    # 'SequenceLabeling/brown_nouns',
    # 'SequenceLabeling/brown_verbs',
    # 'SequenceLabeling/brown_pronouns',
    # 'SequenceLabeling/brown_adverbs',
    #'RationalizedClassify/short_bank_qualified',
    #'RationalizedClassify/bank_qualified',
    #'RationalizedClassify/evidence_inference',
    #'RationalizedClassify/federal_tax',
    #'RationalizedClassify/short_federal_tax',
    #'RationalizedClassify/interest_frequency',
    #'RationalizedClassify/short_interest_frequency',
    #'RationalizedClassify/aviation',
    #'RationalizedClassify/movie_reviews',
    'RationalizedClassify/mining',
}

# Featurizers to activate
FEATURIZERS = {
    "PlainTextFeaturizer",
    # "IndicoStandard",
    "SpacyGloveFeaturizer",
    # "IndicoFastText",
    # "IndicoSentiment",
    # "IndicoElmo",
    # "IndicoTopics",
    # "IndicoFinance",
    # "IndicoTransformer",
    # "IndicoEmotion",
    # "IndicoFastText",
    # "SpacyCNNFeaturizer",
}

# Experiments to run
EXPERIMENTS = {
    # "FinetuneSequenceLabel",
    # "IndicoSequenceLabel"
    "LRBaselineNonRationalized",
    #"DistReweightedGloveClassifierCV",
    #"RationaleInformedLRCV",
    #"FinetuneSeqBaselineRationalized",
    #"FinetuneClfBaselineNonRationalized",
    #"CRFLogit",
    "BiasedLogit",
#    "LogisticRegressionCV",
#    "KNNCV",
#    "TfidfKNN",
#    "TfidfLogisticRegression",
#    "KCenters",
#    "TfidfKCenters"
    # "SupportVectorMachineCV",
}

# Metrics to compute
METRICS = {
#    "Accuracy",
    "AccuracyRationalized",
    "MacroRocAucRationalized",
#    "MacroRocAuc"
}

# Test setup metadata
TEST_SETUP = {
    "train_sizes": [20], #, 40, 60, 80, 100, 150, 200, 300, 400, 500],
    "n_splits": 5,
    "samplers": ['RandomRationalized'],
#    "samplers": ["ImbalanceSampler"],
    "sampling_size": 0.2,
    "resamplers": ['RandomOverSampler']
#    "resamplers": ["RandomOverSampler"],
}

# Visualizations to display
VISUALIZATIONS = {"FacetGridVisualizer"}

# kwargs to pass directly into visualizations
VISUALIZATION_OPTIONS = {
    "display": True,
    "save": True,
    "FacetGridVisualizer": {
        "x_tile": "Metric",
        "y_tile": "Dataset",
        "x_axis": "TrainSize",
        "y_axis": "Result",
        "lines": ["Experiment", "Featurizer", "Sampler", "Resampler"],
        "category": "merge",
        "cv": "mean",
        "filename": "TestResult",
    },
}

MODE = ModeKeys.RATIONALIZED

N_GPUS = 1
N_CORES = 1  # multiprocessing.cpu_count()

FIX_REQUIREMENTS = True

GOLD_FRAC = 0.05
CORRUPTION_FRAC = 0.4

indicoio.config.api_key = ""
