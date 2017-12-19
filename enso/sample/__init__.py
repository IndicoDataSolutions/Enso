import numpy as np
import logging
from sklearn.metrics.pairwise import pairwise_distances


def sample(sampler, data, train_indices, train_size):
    sampler = Sampler._class_for(sampler)(data, train_indices, train_size)
    return sampler.sample()


class Sampler(object):
    """Base class for all samples."""

    def __init__(self, data, train_indices, train_size):
        self.data = data
        self.train_indices = train_indices
        self.train_size = train_size

    def sample(self):
        raise NotImplementedError

    @classmethod
    def _class_for(cls, sampler_string):
        from enso.sample.kcenter_sampler import KCenter
        from enso.sample.random_sampler import Random
        for subclass in cls.__subclasses__():
            if subclass.__name__ == sampler_string:
                return subclass
        raise ValueError("Invalid sampler attempted ({})".format(sampler_string))