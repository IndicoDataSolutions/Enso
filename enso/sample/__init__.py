class Sampler(object):
    """Base class for all samples."""

    def __init__(self, data_points, train_size):
        self.data_points = data_points
        self.train_size = train_size

    def sample(self):
        raise NotImplementedError

    @classmethod
    def class_for(cls, sampler_string):
        for subclass in cls.__subclasses__():
            if subclass.__name__ == sampler_string:
                return subclass
        raise ValueError("Invalid sampler attempted ({})".format(sampler_string))


class Random(Sampler):
    def sample(self):
        return self.data_points[0:self.train_size]


class RandomA(Sampler):
    def sample(self):
        return self.data_points[(len(self.data_points)-self.train_size):]