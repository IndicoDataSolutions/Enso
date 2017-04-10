"""Module for all featurization methods that one might want to test."""


class Featurizer(object):
    """Base class for building featurizers."""

    pass


class UnsupervisedFeaturizer(Featurizer):
    """Base class for Unsupervised Featurizers."""

    pass


class SupervisedFeaturizer(Featurizer):
    """Base class for Supervised Featurizers."""

    pass
