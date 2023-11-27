import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from joblib import dump, load

mbti = pd.read_csv('mbti_all.csv')

mbti_pred = mbti.sample(frac=0.1)

def get_final_words():
    words = list()
    for i in list(mbti_pred['posts']):
        for j in i.split(' '):
            words.append(j)
        
    words_dic = Counter(words)
    words_dic = pd.DataFrame({'Word':list(words_dic.keys()),'Frequency':list(words_dic.values())})
    words_dic.sort_values('Frequency',ascending=False,inplace=True)
    words_dic.set_index('Word',inplace=True)

    words_dic.to_csv('words_select.csv')

    quan_max = words_dic['Frequency'].quantile(0.99)
    words_dic = words_dic[words_dic.Frequency>quan_max]
    final_words = list(words_dic.index)
    
    return final_words

final_words = get_final_words()

def make_mbti_dataset_df():
    for i in final_words:
        mbti[i] = mbti['posts'].apply(lambda x: 1 if  i in x.split(' ') else 0)
    mbti.drop(['Length','posts'],axis=1,inplace=True)
        
    mbti.to_csv("MBTI_words.csv")
    return mbti

def train_model(mbti: pd.DataFrame):
    Personalities = ['ISFP', 'INFP','INFJ','INTP','INT J','ENTP','ENFP','ISTP','ENTJ','ISTJ','ENFJ','ISFJ','ESTP','ESFP','ESFJ','ESTJ']
    
    for i in Personalities:
        temp = mbti.copy()
        temp['Personality Type'] = temp['Personality Type'].apply(lambda x: 1 if x==i else 0)
        X = temp.drop('Personality Type',axis=1)
        y = temp['Personality Type']
        X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7,random_state=100)
        Model = RandomForestClassifier(n_jobs=-1,n_estimators=100, random_state=42, class_weight='balanced')
    
        params = {'n_estimators':[100],
              'max_depth':[3,5,7,10,12,15],
              'max_features':[0.05,0.1,0.15,0.2],
              'criterion':["gini","entropy"]}
    
        grid_search = GridSearchCV(estimator=Model,param_grid=params,verbose=1,n_jobs=-1,scoring='accuracy')
        grid_search.fit(X_train,y_train)
    
        Model_best = grid_search.best_estimator_
    
        dump(Model_best, 'model_common_{0}.joblib'.format(i))