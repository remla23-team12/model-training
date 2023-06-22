"""
This module tests model development.
"""
import joblib
import pickle
import pytest
from ..models import train_model
from ..models import predict_model
from ..features import build_features


@pytest.fixture(name="model")
def load_model():
    """
    Load the trained model.
    """
    classifier = joblib.load("models/classifier_sentiment_model")
    return classifier

@pytest.fixture(name="dataset")
def load_dataset():
    """
    Load the dataset.
    """
    with open("output/splitData/x_data.pkl", "rb") as file:
        x_train = pickle.load(file)
    return x_train

@pytest.fixture(name="groundtruth")
def load_groundTruth():
    """
    Load the ground truth data, consists of 0 or 1s.
    """
    with open("output/splitData/y_data.pkl", "rb") as file:
        y_train = pickle.load(file)
    return y_train

def test_review_feature_memory_usage(dataset):
    """
    Test for Features and Data: memory cost of the review feature
    """
    review_feature_in_bytes = dataset.nbytes
    max_bytes = 1000000000 # equal to 1 GB
    assert (
        review_feature_in_bytes < max_bytes
    ), f"The review feature is using too many bytes: {review_feature_in_bytes}"