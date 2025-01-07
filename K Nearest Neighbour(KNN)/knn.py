import numpy as np
from collections import Counter


def euclidian_distance(x1, x2):
    """
    Calculate the Euclidian distance between two vectors.

    Parameters
    ----------
    x1 : array-like
        The first vector.
    x2 : array-like
        The second vector.

    Returns
    -------
    float
        The Euclidian distance between the vectors.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        """
        Fit the KNN model using the training data.

        Parameters
        ----------
        X : array-like
            The training samples.
        y : array-like
            The class labels for the training samples.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predict the class labels of a set of samples.

        Parameters
        ----------
        X : array-like
            The samples to predict.

        Returns
        -------
        list
            The predicted class labels.
        """
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # compute the distance
        """
        Predict the class label of a sample.

        Parameters
        ----------
        x : array-like
            The sample to predict.

        Returns
        -------
        list
            The most common class label of the k-nearest neighbors.
        """
        distances = [euclidian_distance(x, x_train) for x_train in self.X_train]

        # get the closest k
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority votes
        most_common = Counter(k_nearest_labels).most_common()
        return most_common
