import joblib
import preprocess
import train
import pickle

if __name__ == '__main__':
    preprocess.test_preprocess()
    train.test_train()

    classifier = joblib.load('models/classifier_sentiment_model')
    cv = pickle.load(open('bow/bow_sentiment_model.pkl', 'rb'))

    review = 'We are so glad we found this place.'
    print('review: ', review)
    print('model result: ', classifier.predict(cv.transform([preprocess.preprocess_review(review)]).toarray())[0])

