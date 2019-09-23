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
EXPERIMENT_NAME = "roBERTa"

# Name of the csv used to store results
RESULTS_CSV_NAME = "Results.csv"

# Datasets to featurize or run experiments on
DATA = {
    'Classify/AirlineSentiment',
    'Classify/AirlineNegativity',
    'Classify/BrandEmotion',
    'Classify/BrandEmotionCause',
    'Classify/ChemicalDiseaseCauses',
    'Classify/CorporateMessaging',
    'Classify/CustomerReviews',
#    'Classify/DetailedEmotion',
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
    'Classify/SocialMediaDisasters',
    'Classify/SST-binary',
    'Classify/Subjectivity'
}

# Featurizers to activate
FEATURIZERS = {
    "PlainTextFeaturizer",
#    "GPCClfTokFeaturizer",
#    "GPCFinalStateFeaturizer",
#    "GPCMeanStateFeaturizer",
#    "GPCMaxStateFeaturizer",
#    "GPCMeanTokFeaturizer",
#    "GPCMaxTokFeaturizer",
     "IndicoStandard",
#    "SpacyGloveFeaturizer",
#    "IndicoFastText",
#    "IndicoSentiment",
    "IndicoElmo",
#    "IndicoTopics",
#    "IndicoFinance",
#    "IndicoTransformer",
#    "IndicoEmotion",
#    "IndicoFastText",
#    "SpacyCNNFeaturizer",
}

# Experiments to run
EXPERIMENTS = {
    # "FinetuneSequenceLabel",
    # "IndicoSequenceLabel"
    #"LogisticRegressionCV",
    "SupportVectorMachineCV",
    "FinetuneGPT",
    "FinetuneRoBERTa",
    "FinetuneGPTSummaries",
#    "FinetuneGPTAdaptors",
    "FinetunGPT2",
    "FinetuneBERT",
    "FinetuneGPC",
    "FinetuneDistilBERT"
#    "FinetuneBERTLarge",
#    "FinetuneGPC",
#    "FinetuneGPCPrefit"
}

# Metrics to compute
METRICS = {
    "Accuracy",
    "MacroRocAuc",
}

# Test setup metadata
TEST_SETUP = {
    "train_sizes": list(range(50, 500, 50)),
    "n_splits": 3,
    # "samplers": ['RandomSequence', 'NoSampler'],
    "samplers": ['Random'],
    "sampling_size": .3,
    # "resamplers": ["SequenceOverSampler", 'NoResampler']
    "resamplers": ["RandomOverSampler"]
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
        'lines': ['Experiment', 'Featurizer', "Sampler", "Resampler"],
        'category': 'merge',
        'cv': 'mean',
        'filename': 'TestResult'
    }
}

MODE = ModeKeys.CLASSIFY

N_GPUS = 1
N_CORES = 1 # multiprocessing.cpu_count()

FIX_REQUIREMENTS = True

GOLD_FRAC = 0.05
CORRUPTION_FRAC = 0.4

indicoio.config.api_key = "6ce18bc8af2ad17432b913b05d14bfbd"
