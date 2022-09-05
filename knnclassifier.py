import numpy as np


class KNNClassifier:
    def __init__(self, k: int):
        self.k: int = k
        self.distances = []
        self.features = [[]]
        self.labels = []

    def fit(self, x: [], y: [[]], labels: []):
        self.labels = labels
        self.features = y
        self.distances = [np.linalg.norm(np.array(x) - np.array(y[i])) for i in range(len(self.features) - 1)]

    def predict(self):
        arr_distances = np.array(self.distances)
        k_smallest_values = np.sort(arr_distances)[:self.k]
        smallest_val = np.min(k_smallest_values)
        predicted_index = self.distances.index(smallest_val)
        return self.features[predicted_index], self.labels[predicted_index], self.distances[predicted_index]
