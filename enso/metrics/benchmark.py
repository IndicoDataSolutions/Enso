from collections import defaultdict

import pandas as pd
import numpy as np

from enso.config import EXPERIMENT_NAME, RESULTS_DIRECTORY
from enso.metrics.basic_classification import roc_auc_score

# Settings required: 10 trials, 50 --> 500 examples in increments of 50
# TEST_SETUP = {
#     "train_sizes": list(range(50, 550, 50)),
#     "n_splits": 10,
#     "samplers": ['Random'],
#     "sampling_size": .3,
#     "resamplers": ['RandomOverSampler'] # optional
# }

TRAIN_SIZES = list(range(50, 550, 50))
N_TRIALS = 10
# Datasets required:
REQUIRED_DATASETS = [
    "Classify/{}".format(dataset) for dataset in [
        'AirlineSentiment',
        'AirlineNegativity',
        'BrandEmotion',
        'BrandEmotionCause',
        'ChemicalDiseaseCauses',
        'CorporateMessaging',
        'CustomerReviews',
        'DetailedEmotion',
        'DrugReviewType',
        'DrugReviewIntent',
        'Economy',
        'Emotion',
        'GlobalWarming',
        'MovieReviews',
        'MPQA',
        'NewYearsResolutions',
        'PoliticalTweetBias',
        'PoliticalTweetClassification',
        'SocialMediaDisasters',
        'SST-binary',
        'Subjectivity'
    ]
]

if __name__ == "__main__":
    filename = '{}/{}/Results.csv'.format(RESULTS_DIRECTORY, EXPERIMENT_NAME)
    df = pd.read_csv(filename)
    df = df[df.Metric == 'MacroRocAuc']
    df['Config'] = df['Featurizer'] + df['Experiment']
    datasets = df.groupby('Dataset')
    aggregate_scores = defaultdict(list)

    for config_name, config_df in df.groupby('Config'):
        missing_datasets = set(REQUIRED_DATASETS) - set(config_df.Dataset.unique().tolist())
        missing_datasets = sorted(list(missing_datasets))
        if missing_datasets:
            print("{} missing results for the following datasets:\n\t -{}\n".format(config_name, "\n\t- ".join(missing_datasets)))
            continue
        else:
            for dataset_name, dataset_df in config_df.groupby('Dataset'):
                missing_runs = sorted(list(
                    set(TRAIN_SIZES) - set(np.unique(dataset_df.TrainSize.values.astype(np.int32)).tolist())
                ))
                if missing_runs:
                    print("{} evalution on {} missing results for the following training sizes: {}\n".format(
                        config_name, dataset_name, ", ".join(map(str, missing_runs))
                    ))
                    continue
                n_trials = dataset_df.Result.values.shape[0]
                total_trials = N_TRIALS * len(TRAIN_SIZES)
                if not n_trials == total_trials:
                    print("{} evaluation on {} has invalid number of trials: {} expected, {} observed.\n".format(
                        config_name, dataset_name, total_trials, n_trials
                    ))
                    continue
                aggregate_scores[config_name].append(np.mean(dataset_df.Result))

    for config, scores_arr in aggregate_scores.items():
        print("{}: {}".format(config, np.mean(scores_arr)))
