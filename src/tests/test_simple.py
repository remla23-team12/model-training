"""
This module do a simple test on the model.
"""
import pytest
from ..models import train_model
from ..models import predict_model
from ..features import build_features

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
            accuracy_diff < 0.12
        ), f"Difference in model accuracy with different seed(={seed}) is too high: {accuracy_diff}"
