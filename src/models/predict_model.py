import joblib
from sklearn.metrics import accuracy_score, classification_report
import pickle
import json


def save_metrics(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f)


def test():
    # model-training before the path if running directly, the current path is for dvc
    x_test = pickle.load(open("output/splitData/x_test.pkl", "rb"))
    y_test = pickle.load(open("output/splitData/y_test.pkl", "rb"))
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
