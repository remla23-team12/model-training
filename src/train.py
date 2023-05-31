import pickle
from sklearn.naive_bayes import GaussianNB
import joblib


def train(var_smoothing=1e-09, random_state=None):
    # model-training before the path if running directly, the current path is for dvc
    trail = '' if random_state is None else random_state
    x_train = pickle.load(open(f'splitData/x_train{"_"+str(trail)}.pkl', "rb"))
    y_train = pickle.load(open(f'splitData/y_train{"_"+str(trail)}.pkl', "rb"))

    classifier = GaussianNB(var_smoothing=var_smoothing)
    classifier.fit(x_train, y_train)

    joblib.dump(classifier, f'models/classifier_sentiment_model{"_"+str(trail)}')
    return classifier


def test_train(var_smoothing, random_state):
    train(var_smoothing=var_smoothing, random_state=random_state)


if __name__ == "__main__":
    test_train(1e-010)
