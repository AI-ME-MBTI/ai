from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

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
    