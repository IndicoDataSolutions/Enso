from collections import Counter, defaultdict
import random
from abc import ABCMeta, abstractmethod
import itertools
from enso.config import CORRUPTION_FRAC, GOLD_FRAC

import numpy as np

from enso.registry import Registry, ModeKeys


class Resampler(metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def resample(X, y, max_ratio=50):
        """ """


@Registry.register_resampler(ModeKeys.CLASSIFY)
class RandomOverSampler(Resampler):

    @staticmethod
    def resample(X, y, max_ratio=50):
        X, y = np.asarray(X), np.asarray(y)
        class_counts = Counter(y)
        max_count = max(class_counts.values())
        desired_counts = {
            class_name: min(max_count, max_ratio * class_count) - class_count
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

        return X.tolist() + X[idx_sample].tolist(), y.tolist() + y[idx_sample].tolist()


@Registry.register_resampler(ModeKeys.CLASSIFY)
class CorruptiveResampler(Resampler):

    @staticmethod
    def scramble(y, prob):
        classes = list(set(y))
        y_ = []
        baseline_probas = np.ones(len(classes)) * prob / (len(classes) - 1)
        true_proba = 1 - sum(baseline_probas)
        for sample in y:
            bl = np.copy(baseline_probas)
            bl[classes.index(sample)] += true_proba
            y_.append(np.random.choice(classes, p=bl))
        return y_

    @staticmethod
    def resample(X, y, max_ratio=50):
        num_samples = len(y)
        num_gold = int(num_samples * GOLD_FRAC)

        y_gold = y[:num_gold]
        y_train = np.asarray(list(y_gold) + CorruptiveResampler.scramble(y[num_gold:], CORRUPTION_FRAC))
        return X, y_train


@Registry.register_resampler(ModeKeys.CLASSIFY)
class CorruptiveOversampler(Resampler):

    @staticmethod
    def resample(X, y, max_ratio=50):
        X, y = CorruptiveResampler.resample(X, y)
        return RandomOverSampler.resample(X, y, max_ratio=max_ratio)


@Registry.register_resampler(ModeKeys.ANY)
class NoResampler(Resampler):

    @staticmethod
    def resample(X, y, max_ratio=50):
        return X, y


@Registry.register_resampler(ModeKeys.SEQUENCE)
class SequenceOverSampler(Resampler):

    @staticmethod
    def resample(X, y, max_ratio=50):

        stripped_labels = []
        for item in y:
            per_sample = []
            for label in item:
                per_sample.append(label["label"])
            stripped_labels.append(per_sample)

        class_counts = Counter(itertools.chain.from_iterable(stripped_labels))
        max_count = max(class_counts.values())
        classes = class_counts.keys()

        # Set the objective of the number to add
        need_to_add = [
            min(max_count, max_ratio * class_count) - class_count
            for class_count in class_counts.values()
        ]

        extra_idxs = []
        starting_len = 0
        ending_len = -1
        last_idxs = []

        # While improvement is made
        while starting_len != ending_len:
            starting_len = len(extra_idxs)

            # Argshuffle
            args_shuffled = np.arange(len(y))
            np.random.shuffle(args_shuffled)

            num_to_decrement = [0 for _ in range(len(classes))]
            args_to_add = []

            # Shuffle the args and pull them off one by one seeing if they will fit into the objective.
            for arg in args_shuffled:
                labels_to_add = stripped_labels[arg]
                if len(labels_to_add) == 0:
                    continue
                acceptable = []
                num_to_decrememt_for_sample = []
                for cls_idx, cls in enumerate(classes):
                    cls_count = labels_to_add.count(cls)
                    num_to_decrememt_for_sample.append(cls_count)

                    # Check that the number of instances of each class in this sample do not overflow the requirements
                    acceptable.append(cls_count <= (need_to_add[cls_idx] - num_to_decrement[cls_idx]))
                if all(acceptable):
                    num_to_decrement = [n + n_local for n, n_local in
                                        zip(num_to_decrement, num_to_decrememt_for_sample)]
                    args_to_add.append(arg)

            # If the randomly selected indexes are the same as last time
            # assume these are the only ones that'll fit and add as many as you can.
            while all([n >= h for n, h in zip(need_to_add, num_to_decrement)]):
                extra_idxs.extend(args_to_add)
                need_to_add = [n - h for n, h in zip(need_to_add, num_to_decrement)]
                if last_idxs != sorted(args_to_add):
                    break

            last_idxs = sorted(args_to_add)
            ending_len = len(extra_idxs)

        return X + [X[i] for i in extra_idxs], y + [y[i] for i in extra_idxs]