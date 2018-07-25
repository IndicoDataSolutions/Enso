from collections import Counter, defaultdict
import random
import itertools
from copy import deepcopy


import numpy as np



class OverSampler():


    def oversample_classes(self, X, y, max_ratio=50):
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

    def oversample_sequences(self, X, y, max_ratio=50):
        stripped_labels = []
        for item in y:
            per_sample = []
            for label in item:
                per_sample.append(label["label"])
            stripped_labels.append(per_sample)

        class_counts = Counter(itertools.chain.from_iterable(stripped_labels))
        max_count = max(class_counts.values())
        classes = class_counts.keys()
        need_to_add = {
            class_name: min(max_count, max_ratio * class_count) - class_count
            for class_name, class_count in class_counts.items()
        }

        label_to_idx_copy = deepcopy(label_to_idx)

        extra_idxs = []
        arg_shuffled = np.random.shuffle(np.arange(len(y)))
        for arg in arg_shuffled:
            labels_to_add = y[arg]
            acceptable = []
            num_to_decrement = []
            for cls in classes:
                cls_count = labels_to_add.count(cls)
                num_to_decrement.append(cls_count)
                acceptable.append(cls_count < need_to_add[cls])
            if all(acceptable):
                # TODO(BEN) Finish this

















def resample(resample_type, train_data, train_labels):
    if resample_type.lower() == 'none':
        return train_data, train_labels
    elif resample_type == 'RandomOverSampler':
        train_data, train_labels = oversample(train_data, train_labels)
        return train_data, train_labels
    else:
        raise Exception("Invalid resample_type: {}".format(resample_type))
