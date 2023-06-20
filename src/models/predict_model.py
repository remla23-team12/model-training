"""
This module tests a trained Naive Bayes model on some preprocessed data.
"""
import pickle
import json
from sklearn.metrics import accuracy_score
import joblib


def save_metrics(metrics, path):
    """
    Saves the metrics in a json file.

    Args:
    metrics (dict): a dictionary containing all the metrics to be saved.
    path (str): path where the json file will be saved.
    """
    with open(path, "w", encoding="utf-8") as file:
        json.dump(metrics, file)


def test():
    """
    Tests a pre-trained Gaussian Naive Bayes model on testing data.

    Returns:
    float: accuracy of the model on the testing data.
    """
    # model-training before the path if running directly, the current path is for dvc
    with open("output/splitData/x_test.pkl", "rb") as file:
        x_test = pickle.load(file)
    with open("output/splitData/y_test.pkl", "rb") as file:
        y_test = pickle.load(file)

    classifier = joblib.load("models/classifier_sentiment_model")
    y_pred = classifier.predict(x_test)

    # Save accuracy to a JSON file
    save_metrics(
        {"accuracy": accuracy_score(y_test, y_pred)},
        "output/res/metrics.json",
    )
    return accuracy_score(y_test, y_pred)


if __name__ == "__main__":
    test()
