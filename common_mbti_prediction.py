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
