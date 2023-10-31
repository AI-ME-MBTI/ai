from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from collections import Counter
import pandas as pd


def get_tagged_words(text):
    tagged_list = pos_tag(word_tokenize(text))
    new_tagged_list = []
    
    for w, p in tagged_list:
        new_tagged_list.append([w.lower(), p])
    
    for i in range(len(new_tagged_list)):
        if new_tagged_list[i][1].startswith('V'):
            new_tagged_list[i][1] = 'v'
        elif new_tagged_list[i][1].startswith('N') or new_tagged_list[i][1].startswith('PR'):
            new_tagged_list[i][1] = 'n'
        elif new_tagged_list[i][1].startswith('RB') or new_tagged_list[i][1].startswith('JJ'):
            new_tagged_list[i][1] = 'a'
            
    lm = WordNetLemmatizer()

    filtered_list = [lm.lemmatize(w, pos=p) for w, p in new_tagged_list if p in ['v', 'n', 'a']]
    
    from nltk.corpus import stopwords
    
    stopwords = stopwords.words('english')
    filtered_set = set(filtered_list)
    final_list = filtered_list

    for w in filtered_set:
        if w in stopwords:
            while w in final_list:
                final_list.remove(w)
                
    return final_list

def get_count(final_list):
    count_word = Counter(final_list)
    return count_word

def make_mbti_dataset_csv():
    mbti = pd.read_csv('MBTI 500.csv', encoding='ISO 8859-1', header=None, names=['posts', 'type'])
    
    types = mbti['type'].unique()[1:]
    types.sort()

    for t in types:
        full_post = ' '.join(mbti[mbti['type']==t].posts)
        filtered_list = get_tagged_words(full_post)
        counts = get_count(filtered_list)
        mbti_posts = counts
        
        words_dic = pd.DataFrame({'Word': list(mbti_posts.keys()), 'Frequency': list(mbti_posts.values())})
        words_dic.sort_values('Frequency', ascending=False, inplace=True)
        words_dic.set_index('Word', inplace=True)
        words_dic.to_csv('{}.csv'.format(t))