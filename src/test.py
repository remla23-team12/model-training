import joblib
from sklearn.metrics import accuracy_score, classification_report
import pickle
import json


def save_metrics(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f)


def test(random_state=None):
    # model-training before the path if running directly, the current path is for dvc
    trail = '' if random_state is None else random_state
    x_test = pickle.load(open(f'splitData/x_test{"_"+str(trail)}.pkl', "rb"))
    y_test = pickle.load(open(f'splitData/y_test{"_"+str(trail)}.pkl', "rb"))
    classifier = joblib.load(f'models/classifier_sentiment_model{"_"+str(trail)}')
    y_pred = classifier.predict(x_test)

    # Save accuracy to a JSON file
    save_metrics(
        {"accuracy": accuracy_score(y_test, y_pred)},
        "output/metrics.json",
    )
    return accuracy_score(y_test, y_pred)


if __name__ == "__main__":
    test()
