"""Tests for Training algorithm"""
import math

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from custom_regressor import NewTrainer
from errors import InputError
import numpy as np
import pytest
import random


@pytest.fixture
def sample_data():
    """Supply proper data for testing"""
    X = np.linspace(1, 10, 100)
    Y = 10 * X + 100
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)
    return X, Y


@pytest.fixture
def sample_data_trained(sample_data):
    """Supply proper data for testing"""
    X = sample_data[0]
    Y = sample_data[1]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.80)
    ml = NewTrainer()
    ml.train(x_train, y_train)
    return ml, x_train, x_test, y_train, y_test


def test_input_dim():
    """Test how trainer handle invalid data"""
    with pytest.raises(InputError):
        X = [random.sample(range(80), 10)] * 6
        Y = random.sample(range(80), 10)
        ml = NewTrainer()
        ml.train(X, Y)


def test_input_single():
    """Test if single feature proceeds without error"""
    X = [1] * 10
    Y = [1] * 10
    ml = NewTrainer()
    # noinspection PyTypeChecker
    ml.train(X, Y)


def test_model_generated(sample_data_trained):
    """Test if model is not None"""
    ml = sample_data_trained[0]
    assert isinstance(
        ml.model,
        sklearn.linear_model._base.LinearRegression
    )


def test_model_score(sample_data_trained):
    """
    Test R2 score on test datam, similarly MAE and RMSE
    If classification task we could check for Precission, Recall and Accuracy
    """
    ml = sample_data_trained[0]
    x_test = sample_data_trained[2]
    y_test = sample_data_trained[4]
    score = r2_score(
        y_test,
        ml.model.predict(x_test)
    )
    assert score > 0.7
    
    
def test_model_response(sample_data, sample_data_trained):
    """Test if model behaves as expected for variations in data"""
    ml2 = NewTrainer()
    X = sample_data[0] + 1000
    Y = sample_data[1] + 1000
    ml2.train(X, Y)
    ml = sample_data_trained[0]
    assert math.isclose(ml2.model.coef_, ml.model.coef_)



