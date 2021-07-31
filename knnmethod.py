from abc import ABC, abstractmethod

class KNNMethod(ABC):

    def __init__(self):
        self.is_trained = False
        super().__init__()

    @abstractmethod
    def train(self, training_data):
        pass

    @abstractmethod
    def search(self, test_data, k):
        pass

from sklearn.neighbors import NearestNeighbors

class SklearnKNN(KNNMethod):

    def __init__(self, algorithm):
        self.nbrs = NearestNeighbors(algorithm=algorithm)
        self.is_trained = False
        super().__init__()

    def train(self, training_data):
        self.nbrs.fit(training_data)
        self.is_trained = True

    def search(self, test_data, k):
        return self.nbrs.kneighbors(test_data, k)

import faiss

class FaissKNN(KNNMethod):
    def __init__(self, dimension, factory_string):
        self.nbrs = faiss.index_factory(dimension, factory_string)
        self.is_trained = self.nbrs.is_trained
        super().__init__()

    def train(self, training_data):
        self.nbrs.train(training_data)
        self.nbrs.add(training_data)

    def search(self, test_data, k):
        return self.nbrs.search(test_data, k)