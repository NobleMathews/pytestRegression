"""
A sample training algorithm for regression
"""
import logging

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from typing import List
import numpy

from errors import InputError


class NewTrainer:
    """
    A linear regression based trainer using sklearn
    """
    model = None

    def train(self, x: List[List[float]], y: List[float]):
        """
        Train a linear regressor
        :param x: Training Data
        :param y:
        :return:
        """
        X = numpy.array(x)
        Y = numpy.array(y)

        if X.shape[0] != Y.shape[0]:
            raise InputError(
                f"({X.shape},{Y.shape})"
            )

        # Single feature or sample, add a dimension
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)

        self.model = LinearRegression()
        self.model.fit(X, Y)

    def predict(self, x: List[float]) -> float:
        """
        Predict and log y
        :param x: testing data
        :return: 0
        """
        y = self.model.predict(x)
        logging.info(y)
        return 0
