from numpy.random import choice


class Sampler(object):
    """Base class for all samples."""

    def __init__(self, data, train_indices, train_size):
        self.data = data
        self.train_indices = train_indices
        self.train_size = train_size

    def sample(self):
        raise NotImplementedError

    @classmethod
    def sample(cls, sampler, data, train_indices, train_size):
        return cls._class_for(sampler)(data, train_indices, train_size).sample()
    
    @classmethod
    def _class_for(cls, sampler_string):
        for subclass in cls.__subclasses__():
            if subclass.__name__ == sampler_string:
                return subclass
        raise ValueError("Invalid sampler attempted ({})".format(sampler_string))


class Default(Sampler):
    def sample(self):
        return self.train_indices[0:self.train_size]


class Random(Sampler):
    def sample(self):
        return choice(self.train_indices, self.train_size, replace=False)