from . import Sampler


class KCenter(Sampler):
    @property
    def points(self):
        return [np.array(point) for point in self.data]

    def sample(self):
        random_center = np.random.choice(self.train_indices)
        centers = [random_center]
        while len(centers) < self.train_size:
            mins = np.min(self.distances[centers], axis=0)
            center = np.argmax(mins)
            centers.append(center)
        return centers
    
    @property
    def distances(self):
        if not hasattr(self, '_distances'):
            self._distances = pairwise_distances(self.points)
        return self._distances