import pandas as pd
import os

from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pickle
import joblib

from papago.papago import get_translate

detail_mbti = pd.read_csv('./csv/mbti_detail_data.csv')

def mbti_train_and_prediction(mbti_type: str, answer: str):
    type_to_index = {"IE": 0, "SF": 2, "FT": 4, "PJ": 6}
    
    i = type_to_index[mbti_type]
    mbti = detail_mbti[i:i+2]
    
    X = mbti['word']
    y = mbti['mbti']
    
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X)
    joblib.dump(vectorizer, './models/vectorizer_detail_text.sav')

    X_test = answer

    clf = LinearSVC()
    clf.fit(X_train_tfidf, y)
    pickle.dump(clf, open('./models/model_detail_mbti_{0}.sav'.format(mbti_type), 'wb'))

    y_train_pred = clf.predict(X_train_tfidf)
    print("Train Accuracy:", accuracy_score(y, y_train_pred))

    X_test_tfidf = vectorizer.transform([X_test])
    y_test_pred = clf.predict(X_test_tfidf)
    y_test_pred = ''.join(y_test_pred)
    
    return y_test_pred

def get_specific_mbti(mbti_type: str, answer:str):
    kr_answer = get_translate(answer)
    result = mbti_train_and_prediction(mbti_type, kr_answer)
    return result

def get_feedbackf(user_feedback):
    mbti_name = {"I": "IE", "E": "IE", "S": "SN", "N": "SN", "T": "TF", "F": "TF", "P": "PJ", "J":"PJ"}
    
    for data in user_feedback:
        mbti = data.detail_mbti
        answer = data.answer
        
        if not os.path.exists('./feedback/detail/detail_feedback_{0}.csv'.format(mbti_name[mbti])):
            feedback_df = pd.DataFrame({'word': [answer], 'mbti': mbti})
        else:
            feedback_df = pd.read_csv('./feedback/detail/detail_feedback_{0}.csv'.format(mbti_name[mbti]))
            
            feedback_df = feedback_df.append({'word': answer, 'mbti': mbti}, ignore_index=True)
            
    feedback_df.to_csv('./feedback/detail/detail_feedback_{0}.csv'.format(mbti_name[mbti]))

def extra_train_specific_model():
    mbti_type = ['IE', 'SN', 'TF', 'PJ']
    
    for m in mbti_type:
        feedback_df = pd.read_csv('./feedback/detail/detail_feedback_{0}.csv'.format(m))
    
        if len(feedback_df) >= 3:
            X = feedback_df['word']
            y = feedback_df['mbti']
        
            vectorizer = joblib.load('./models/vectorizer_detail_text.sav')
            clf = pickle.load(open('./models/model_detail_mbti_{0}.sav'.format(m), 'rb'))
        
            X_train_tfidf = vectorizer.fit_transform(X)
            clf.fit(X_train_tfidf, y)
        
            joblib.dump(vectorizer, './models/vectorizer_detail_text.joblib')
            pickle.dump(clf, open('./models/model_detail_mbti_{0}.sav'.format(m)), 'wb')
        else:
            return False
        