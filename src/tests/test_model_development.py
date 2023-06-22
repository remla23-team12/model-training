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

@pytest.fixture(name="default_variable_smoothing")
def var_smoothing():
    """
    Yields the amount of smoothing, to be used in some tests for model development tests.
    """
    smoothing = 1.1e-09
    yield smoothing

@pytest.fixture(name="default_score")
def default_score_v1_none(default_variable_smoothing):
    "Yields the score of the model with no random state and 1.1e-09 smoothing"
    build_features.test_preprocess(random_state=None)
    train_model.test_train(default_variable_smoothing)
    score1 = predict_model.test()
    # baseline score
    yield score1

def test_model_diff(default_score):
    """
    Test model development: fails if difference is score caused by different variable smoothing
    is too large
    """
    var_smoothing2 = 1.2e-09
    train_model.test_train(var_smoothing2)
    score2 = predict_model.test()
    accuracy_diff = abs(default_score - score2)
    assert (
        accuracy_diff < 0.1
    ), f"Difference in model accuracy with different seeds is too high: {accuracy_diff}"

def test_nondeterminism_robustness(default_score, default_variable_smoothing):
    """
    Test model development: fails if the score of the model with a different seed is too high
    """
    for seed in [1, 2, 3, 4, 5, 42]:
        build_features.test_preprocess(random_state=seed)
        train_model.test_train(default_variable_smoothing)
        score42 = predict_model.test()
        accuracy_diff = abs(default_score - score42)
        assert (
            accuracy_diff < 0.2
        ), f"Difference in model accuracy with different seed(={seed}) is too high: {accuracy_diff}"

def test_positive_negative_data_slices(model, dataset, groundtruth):
    """
    Test model development: testing the model using data slices.
    The 2 data slices are: 
    - positive data slice (i.e. groundtruth is 1)
    - negative data slice (i.e. groundtruth is 0)
    """
    positive_slice = [dataset[i] for i in range(len(groundtruth)) if groundtruth[i]]
    negative_slice = [dataset[i] for i in range(len(groundtruth)) if not groundtruth[i]]

    positive_score = sum(model.predict(positive_slice) == 1) / len(positive_slice)
    negative_score = sum(model.predict(negative_slice) == 0) / len(negative_slice)
    accuracy_diff = abs(positive_score - negative_score)

    assert (
        accuracy_diff < 0.2
    ), f"Difference in model accuracy with positive and negative slices is too high: {accuracy_diff}"

def test_review_feature_memory_usage(dataset):
    """
    Test for Features and Data: memory cost of the review feature
    """
    review_feature_in_bytes = dataset.nbytes
    max_bytes = 1000000000 # equal to 1 GB
    assert (
        review_feature_in_bytes < max_bytes
    ), f"The review feature is using too many bytes: {review_feature_in_bytes}"