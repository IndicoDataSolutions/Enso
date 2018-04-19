from collections import Counter, defaultdict
import random

import numpy as np


def oversample(X, y, max_ratio=50):
    """
    Ensure each class occurs with approximately even frequency in the training set by
    duplicating examples from relatively rare classes.

    :param X: `np.ndarray` of input features
    :param y: `np.ndarray` of corresponding targets
    :param max_ratio: input examples should be duplicated no more than this amount
    """
    X, y = np.asarray(X), np.asarray(y)
    class_counts = Counter(y)
    max_count = max(class_counts.values())
    desired_counts = {
        class_name: min(
            max_count, max_ratio * class_count)
        for class_name, class_count in class_counts.items()
    }

    idxs_by_y = defaultdict(list)
    for i, element in enumerate(y):
        idxs_by_y[element].append(i)

    idx_sample = []
    for class_name in class_counts:
        sample_idxs = np.random.choice(idxs_by_y[class_name], desired_counts[class_name], replace=True).tolist()
        idx_sample.extend(sample_idxs)

    random.shuffle(idx_sample)

    return X[idx_sample].tolist(), y[idx_sample].tolist()


def resample(resample_type, train_data, train_labels):
    if resample_type.lower() == 'none':
        return train_data, train_labels
    elif resample_type == 'RandomOverSampler':
        train_data, train_labels = oversample(train_data, train_labels)
        return train_data, train_labels
    else:
        raise Exception("Invalid resample_type: {}".format(resample_type))
