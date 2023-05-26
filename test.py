import joblib
import preprocess
import train_bow_gnb
import train_bow_svm
import train_tf_idf_gnb
import train_tf_idf_svm
import pickle

if __name__ == '__main__':
    preprocess.test_preprocess()
    train_bow_gnb.test_train()
    train_bow_svm.test_train()
    train_tf_idf_gnb.test_train()
    train_tf_idf_svm.test_train()

    bow_gnb_classifier = joblib.load('models/bow_gnb_classifier_sentiment_model')
    bow_gnb = pickle.load(open('bow/bow_gnb_sentiment_model.pkl', 'rb'))

    bow_svm_classifier = joblib.load('models/bow_svm_classifier_sentiment_model')
    bow_svm = pickle.load(open('bow/bow_svm_sentiment_model.pkl', 'rb'))

    tfidf_gnb_classifier = joblib.load('models/tfidf_gnb_classifier_sentiment_model')
    tfidf_gnb = pickle.load(open('tfidf/tfidf_gnb_sentiment_model.pkl', 'rb'))

    tfidf_svm_classifier = joblib.load('models/tfidf_svm_classifier_sentiment_model')
    tfidf_svm = pickle.load(open('tfidf/tfidf_svm_sentiment_model.pkl', 'rb'))

    review = 'like'
    print('review: ', review)
    print('bow gnb model result: ', bow_gnb_classifier.predict(bow_gnb.transform([preprocess.preprocess_review(review)]).toarray())[0])
    print('bow svm model result: ', bow_svm_classifier.predict(bow_svm.transform([preprocess.preprocess_review(review)]).toarray())[0])
    print('tfidf gnb model result: ', tfidf_gnb_classifier.predict(tfidf_gnb.transform([preprocess.preprocess_review(review)]).toarray())[0])
    print('tfidf svm model result: ', tfidf_svm_classifier.predict(tfidf_svm.transform([preprocess.preprocess_review(review)]).toarray())[0])


