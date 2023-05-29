"""
This module trains a Naive Bayes model on some preprocessed data.
"""
import pickle
from sklearn.naive_bayes import GaussianNB
import joblib


def train(var_smoothing=1e-09):
    """
    Train a Gaussian Naive Bayes model on training data.

    Args:
    var_smoothing (float): portion of the largest variance of all
    features that is added to variances for calculation stability.

    Returns:
    GaussianNB: the trained model.
    """
    # model-training before the path if running directly, the current path is for dvc
    with open("output/splitData/x_train.pkl", "rb") as file:
        x_train = pickle.load(file)
    with open("output/splitData/y_train.pkl", "rb") as file:
        y_train = pickle.load(file)

    classifier = GaussianNB(var_smoothing=var_smoothing)
    classifier.fit(x_train, y_train)

    joblib.dump(classifier, "models/classifier_sentiment_model")
    return classifier


def test_train(var_smoothing):
    """
    Function to test the train function by invoking it.

    Args:
    var_smoothing (float): portion of the largest variance of all features
    that is added to variances for calculation stability.
    """
    train(var_smoothing=var_smoothing)


if __name__ == "__main__":
    test_train(1e-09)
