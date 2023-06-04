"""
This module preprocesses data, applies bag of words model,
and splits data for model training and testing.
"""
import os
import re
import pickle
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

stopwords = stopwords.words("english")
stopwords.remove("not")

ps = PorterStemmer()


def preprocess_review(review):
    """
    Preprocesses a review by removing non alphabetic characters,
    changing to lower case, splitting into words, and stemming.
    Args:
    review (str): A review text.

    Returns:
    str: A preprocessed review.
    """
    review = re.sub("[^a-zA-Z]", " ", review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords)]
    review = " ".join(review)
    return review


def preprocess_data(filename, random_state=None):
    """
    Preprocesses data, applies bag of words model, and splits data for model training and testing.

    Args:
    filename (str): Filename of the raw data.

    Returns:
    tuple: A tuple containing split data (x_train, x_test, y_train, y_test)
    """
    filepath = f"data/raw/{filename}"
    if not os.path.exists(filepath):
        print(f"{filename} does not exist")
        return None

    dataset = pd.read_csv(filepath, delimiter="\t", quoting=3)
    dataset["Review"] = dataset["Review"].apply(preprocess_review)
    count_vectorizer = CountVectorizer(max_features=1420)
    x_data = count_vectorizer.fit_transform(dataset["Review"].tolist()).toarray()
    y_data = dataset.iloc[:, -1].values

    bow_path = "output/bow/bow_sentiment_model.pkl"
    with open(bow_path, "wb") as file:
        pickle.dump(count_vectorizer, file)

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.20, random_state=random_state
    )

    trail = '' if random_state is None else random_state

    with open(f"output/splitData/x_train{'_'+str(trail)}.pkl", "wb") as file:
        pickle.dump(x_train, file)
    with open(f"output/splitData/x_test{'_'+str(trail)}.pkl", "wb") as file:
        pickle.dump(x_test, file)
    with open(f"output/splitData/y_train{'_'+str(trail)}.pkl", "wb") as file:
        pickle.dump(y_train, file)
    with open(f"output/splitData/y_test{'_'+str(trail)}.pkl", "wb") as file:
        pickle.dump(y_test, file)

    return x_train, x_test, y_train, y_test


def test_preprocess(filename="a1_RestaurantReviews_HistoricDump.tsv", random_state=None):
    """
    Function to test the preprocess_data function by invoking it.

    Args:
    filename (str): Filename of the raw data.
    """
    preprocess_data(filename=filename, random_state=random_state)


if __name__ == "__main__":
    test_preprocess("a1_RestaurantReviews_HistoricDump.tsv")
