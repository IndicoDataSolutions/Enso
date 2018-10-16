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
EXPERIMENT_NAME = "FinetuneMini2"

# Datasets to featurize or run experiments on
DATA = {
    #"Classify/AirlineComplaints",
    "Classify/AirlineNegativity",
    # "Classify/AirlineSentiment",
    "Classify/BrandEmotion",
    # "Classify/BrandEmotionCause",
    # "Classify/ChemicalDiseaseCauses",
    # "Classify/CorporateMessaging",
    #"Classify/CustomerReviews",
    # "Classify/DetailedEmotion",
    # "Classify/Disaster",
    # "Classify/DrugReviewIntent",
    # "Classify/DrugReviewType",
    # "Classify/Economy",
    # "Classify/Emotion",
    # "Classify/GlobalWarming",
    # "Classify/Horror",
    # "Classify/HotelReviews",
     "Classify/IMDB",
     "Classify/Irony",
     "Classify/MPQA",
     "Classify/MovieReviews",
    # "Classify/NewYearsResolutions",
    # "Classify/PoliticalTweetAlignment",
    # "Classify/PoliticalTweetBias",
    # "Classify/PoliticalTweetClassification",
    # "Classify/PoliticalTweetSubjectivity",
    # "Classify/PoliticalTweetTarget",
    # "Classify/Reddit.10cls.1000",
    # "Classify/Reddit.20cls.500",
    # "Classify/Reddit.50cls.200",
    # "Classify/Reddit.5cls.1000",
    # "Classify/ReligiousTexts",
    # "Classify/ShortAnswer",
     "Classify/SocialMediaDisasters",
     #"Classify/Subjectivity",
      "Classify/TextSpam",
    "Classify/SST-binary"

    # Seqence

    # 'SequenceLabeling/Reuters-128',
    # 'SequenceLabeling/brown_all',
    # 'SequenceLabeling/brown_nouns',
    # 'SequenceLabeling/brown_verbs',
    # 'SequenceLabeling/brown_pronouns',
    # 'SequenceLabeling/brown_adverbs',
}

# Featurizers to activate
FEATURIZERS = {
    "PlainTextFeaturizer",
    # "IndicoStandard",
#    "SpacyGloveFeaturizer",
#    "IndicoStandard",
#    "IndicoFastText",
#    "IndicoSentiment",
   # "IndicoElmo",
#    "Word2Vec",
#    "BayesianesqueFeaturizer"
    #"IndicoTopics",
#    "IndicoFinance",
   # "IndicoTransformer",
#    "IndicoEmotion",
#    "IndicoFastText",
    #"IndicoElmo",
#    "SpacyCNNFeaturizer",

}

# Experiments to run
EXPERIMENTS = {
    # "FinetuneSequenceLabel",
    # "IndicoSequenceLabel"
    #"Finetune",
    # "SpacyGlove"
#    "LogisticRegressionCV",
#    "SupportVectorMachineCV",
    #"ReweightingLR",
#    "TfidfLogisticRegression"
     "Finetune2Layers",
    "Finetune2LayersReClf",
    "Finetune"
    # "Finetune4Layers",
    # "Finetune4LayersCV",
    # "Finetune6Layers",
    # "Finetune8Layers",
    # "Finetune10Layers",
    # "Finetune2",
    # "Finetune2Summative",
    # "Finetune2Mean",
    # "Finetune4",
    # "Finetune6",
    # "Finetune8",
    # "Finetune10"
    # "Finetune2CV"
    # "FinetuneLast2CV",
    # "Finetune2CVRoc"
    # "Finetune8CV",
    # "FinetuneLast8CV",
    #"FinetuneCVNumLayers"

}

# Metrics to compute
METRICS = {
    # "OverlapAccuracy",
    # "OverlapPrecision",
    # "OverlapRecall",
    "Accuracy",
    "MacroRocAuc",
}

# Test setup metadata
TEST_SETUP = {
    "train_sizes": range(50, 550, 25),
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
N_CORES = 1  # multiprocessing.cpu_count()

FIX_REQUIREMENTS = True

GOLD_FRAC = 0.05
CORRUPTION_FRAC = 0.4

indicoio.config.api_key = "6ce18bc8af2ad17432b913b05d14bfbd"
