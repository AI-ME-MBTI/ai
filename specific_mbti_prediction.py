import pandas as pd

detail = pd.read_csv('mbti_detail_final.csv')
all_words = detail['word'].unique()

def make_grouped_mbti():
    grouped = detail.groupby('mbti')['word'].apply(lambda x: ' '.join(x)).reset_index(name='word_list')
    
    for i in all_words:
        grouped[i] = grouped['word_list'].apply(lambda x: 1 if i in x.split() else 0)
    grouped.drop(['word_list'], axis=1, inplace=True)
    
    return grouped
