import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import joblib


def train(filename):
    filepath = f'preprocessed_training_data/{filename}'
    if not os.path.exists(filepath):
        print(filename + ' does not exist')
    dataset = pd.read_csv(filepath, delimiter='\t', quoting=3)
    tfidf = TfidfVectorizer(max_features=1420)
    x = tfidf.fit_transform(dataset['Review'].map(lambda review: str(review)).tolist()).toarray()
    y = dataset.iloc[:, -1].values

    tfidf_path = 'tfidf/tfidf_svm_sentiment_model.pkl'
    pickle.dump(tfidf, open(tfidf_path, 'wb'))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    classifier = LinearSVC()
    classifier.fit(x_train, y_train)

    joblib.dump(classifier, 'models/tfidf_svm_classifier_sentiment_model')


def test_train():
    train('preprocessed_a1_RestaurantReviews_HistoricDump.tsv')