"""
This module do a simple test on the model.
"""
from ..models import train_model
from ..models import predict_model


def test_model_diff():
    """
    Test the model by comparing the accuracy of the model with different seeds.
    """
    var_smoothing1 = 1.1e-09
    var_smoothing2 = 1.2e-09
    train_model.test_train(var_smoothing1)
    score1 = predict_model.test()
    train_model.test_train(var_smoothing2)
    score2 = predict_model.test()
    accuracy_diff = abs(score1 - score2)
    assert (
        accuracy_diff < 0.1
    ), f"Difference in model accuracy with different seeds is too high: {accuracy_diff}"
