import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

nltk.download("stopwords")
ps = PorterStemmer()

all_stopwords = stopwords.words("english")
all_stopwords.remove("not")


def preprocess_data(filename, random_state=None):
    # note: if you run the file directly, you need to have model-training before the path
    filepath = f"training_data/{filename}"
    if not os.path.exists(filepath):
        print(filename + " does not exist")
        return
    dataset = pd.read_csv(filepath, delimiter="\t", quoting=3)
    dataset["Review"] = dataset["Review"].map(lambda review: preprocess_review(review))
    # dataset.to_csv(
    #     f"preprocessed_training_data/preprocessed_{filename}",
    #     sep="\t",
    #     quoting=3,
    #     index=False,
    #     encoding="utf-8",
    # )
    cv = CountVectorizer(max_features=1420)
    x = cv.fit_transform(
        dataset["Review"].map(lambda review: str(review)).tolist()
    ).toarray()
    y = dataset.iloc[:, -1].values

    bow_path = "bow/bow_sentiment_model.pkl"
    pickle.dump(cv, open(bow_path, "wb"))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=random_state
    )
    trail = '' if random_state is None else random_state
    # print()   
    pickle.dump(x_train, open(f'splitData/x_train{"_"+str(trail)}.pkl', "wb"))
    pickle.dump(x_test, open(f'splitData/x_test{"_"+str(trail)}.pkl', "wb"))
    pickle.dump(y_train, open(f'splitData/y_train{"_"+str(trail)}.pkl', "wb"))
    pickle.dump(y_test, open(f'splitData/y_test{"_"+str(trail)}.pkl', "wb"))
    return x_train, x_test, y_train, y_test


def preprocess_review(review):
    review = re.sub("[^a-zA-Z]", " ", review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = " ".join(review)
    return review


def test_preprocess(filename="a1_RestaurantReviews_HistoricDump.tsv", random_state=None):
    preprocess_data(filename=filename, random_state=random_state)


if __name__ == "__main__":
    test_preprocess("a1_RestaurantReviews_HistoricDump.tsv")
