import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from joblib import dump, load

detail = pd.read_csv('mbti_detail_final.csv')
all_words = detail['word'].unique()

def make_grouped_mbti():
    grouped = detail.groupby('mbti')['word'].apply(lambda x: ' '.join(x)).reset_index(name='word_list')
    
    for i in all_words:
        grouped[i] = grouped['word_list'].apply(lambda x: 1 if i in x.split() else 0)
    grouped.drop(['word_list'], axis=1, inplace=True)
    
    return grouped

def train_model():
    grouped = make_grouped_mbti()
    personalities = ['I', 'E',  'S', 'N', 'T', 'F', 'J', 'P']
    
    for i in personalities:
        temp = grouped.copy()
        temp['mbti'] = temp['mbti'].apply(lambda x: 1 if x==i else 0)
        X = temp.drop('mbti',axis=1)
        y = temp['mbti']
    
        model = RandomForestClassifier(n_jobs=-1,n_estimators=100, random_state=42, class_weight='balanced')
    
        params = {'n_estimators':[100],
              'max_depth':[3,5,7,10,12,15],
              'max_features':['auto', 'log2', None],
              'criterion':["gini","entropy"]}
    
        grid_search = GridSearchCV(estimator=model,param_grid=params,verbose=1,n_jobs=-1,scoring='accuracy', cv=StratifiedKFold(n_splits=3))
        
        grid_search.fit(X,y)
    
        Model_best = grid_search.best_estimator_
    
        dump(Model_best, 'model_detail_{0}.joblib'.format(i))
    
# train_model()
    
def user_text_to_datagrame(answer: str):
    text = pd.DataFrame({'Text':answer})

    for i in all_words:
        text[i] = text['Text'].apply(lambda x: 1 if i in x.split(' ') else 0)
    text.drop(['Text'],axis=1,inplace=True)
    return text

def mbti_prediction(mbti_type: list[str], answer: pd.DateOffset):
    for i in mbti_type:
        model_best = load('model_detail_{0}.joblib'.format(i))
        user_pred = model_best.predict(answer)
    
        if user_pred == [1]:
            return i
        
    return ''

def user_text_to_datagrame(answer: str):
    text = pd.DataFrame({'Text':answer})

    for i in all_words:
        text[i] = text['Text'].apply(lambda x: 1 if i in x.split(' ') else 0)
    text.drop(['Text'],axis=1,inplace=True)
    return text