import pickle
from sklearn.naive_bayes import GaussianNB
import joblib


def train(var_smoothing=1e-09):
    # model-training before the path if running directly, the current path is for dvc
    x_train = pickle.load(open("splitData/x_train.pkl", "rb"))
    y_train = pickle.load(open("splitData/y_train.pkl", "rb"))
    classifier = GaussianNB(var_smoothing=var_smoothing)
    classifier.fit(x_train, y_train)

    joblib.dump(classifier, "models/classifier_sentiment_model")
    return classifier


def test_train(var_smoothing):
    train(var_smoothing=var_smoothing)


if __name__ == "__main__":
    test_train(1e-010)
