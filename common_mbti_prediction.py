from papago.papago import get_translate

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load
import os

mbti = pd.read_csv('./csv/MBTI_sample.csv')

def train_model():
    try:
        X = mbti['posts']
        y = mbti['type']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        vectorizer = TfidfVectorizer()
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        dump(vectorizer, './models/vectorizer_text.joblib')

        model = RandomForestClassifier(n_jobs=-1,n_estimators=100, random_state=42, class_weight='balanced')

        params = {'n_estimators':[100],
                    'max_depth':[3,5,7,10,12,15],
                    'max_features':[0.05,0.1,0.15,0.2],
                    'criterion':["gini","entropy"]}

        grid_search = GridSearchCV(estimator=model,param_grid=params,verbose=1,n_jobs=-1,scoring='accuracy')
        grid_search.fit(X_train_tfidf,y_train)

        model_best = grid_search.best_estimator_

        y_train_pred = model_best.predict(X_train_tfidf)
        y_test_pred = model_best.predict(X_test_tfidf)

        dump(grid_search, './models/model_common_mbti.joblib')

        print('Train Accuracy:', accuracy_score(y_train, y_train_pred))
        print('Test Accuracy :',accuracy_score(y_test,y_test_pred))

        return True
    except:
        return False
    
def make_common_feedback_df(user_answer: str, user_mbti: str):
    eng_answer = get_translate(user_answer)
    
    if not os.path.exists('./feedback/common/common_feedback.csv'):
        feedback_df = pd.DataFrame({'posts': [eng_answer], 'type': user_mbti})
    else:
        feedback_df = pd.read_csv('./feedback/common/common_feedback.csv')
        feedback_df = pd.concat([feedback_df, pd.DataFrame({'posts': [eng_answer], 'type': user_mbti})], ignore_index=True)
        
    feedback_df.to_csv('./feedback/common/common_feedback.csv')
    
def extra_train_model():
    if os.path.exists('./feedback/common/common_feedback.csv'):
        feedback_df = pd.read_csv('./feedback/common/common_feedback.csv')
        
        if len(feedback_df) >= 5:
            X = feedback_df['posts']
            y = feedback_df['type']
            
            vectorizer = load('./models/vectorizer_text_re.joblib')
            train_tfidf = vectorizer.fit_transform(X)
        
            grid_search = load('./models/model_common_mbti_all.joblib')
            grid_search.fit(train_tfidf, y)
        
            dump(vectorizer, './models/vectorizer_text_1207.joblib')
            dump(grid_search, './models/model_common_mbti_1207.joblib')
            
            return True, len(feedback_df)
        else:
            return False, len(feedback_df)
    
    else:
        return False, 0

def mbti_prediction(answer: str):
    vectorizer = load('./models/vectorizer_text_1207.joblib')
    answer_tfidf = vectorizer.transform([answer])
    
    grid_search = load('./models/model_common_mbti_1207.joblib')
    
    model_best = grid_search.best_estimator_
    user_pred = model_best.predict(answer_tfidf)
    user_pred = ''.join(user_pred)
    
    return user_pred

def get_common_mbti(answer: str):
    kr_answer = get_translate(answer)
    mbti = mbti_prediction(kr_answer)
    return mbti
