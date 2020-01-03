import abc

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


class DimensionReducer(abc.ABC):
    """Base class for certain types of reducers. Implementations need to implement def apply """

    @staticmethod
    @abc.abstractmethod
    def apply(predictions, labels, dimensions: int):
        """
        Apply dimensionalirty reduction
        :param predictions: Predictions to reduce
        :param labels: True labels for predictions
        :param dimensions:  Number of dimensions to reduce to
        """
        pass


class LDA(DimensionReducer):
    """Linear Discriminant Analysis implementation of dimension reduction"""

    @staticmethod
    def apply(predictions, labels, dimensions: int):
        assert len(predictions[0]) > dimensions, \
            f'Cannot reduce to #{dimensions} for data with #{len(predictions[0])}'
        assert len(predictions) == len(labels), \
            f'Nr of predictions ({len(predictions)}) does not match nr of labels ({len(labels)})'
        predictions, labels = LDA._preprocess(predictions, labels)
        lda = LinearDiscriminantAnalysis(n_components=dimensions)
        labels = np.ravel(labels)
        lda.fit(predictions, labels)
        return lda

    @staticmethod
    def _preprocess(predictions, labels):
        predictions = LDA._standardize_data(predictions)
        labels = [np.where(label == 1)[0] for label in labels]
        return predictions, labels

    @staticmethod
    def _standardize_data(predictions):
        return StandardScaler().fit_transform(predictions)
