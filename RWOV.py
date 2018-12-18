
# coding: utf-8

# In[20]:

import pandas as pd
import re  
import csv
import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import sent_tokenize
import string

PATHOLOGY_FILE = 'S:/Biostats/BIO-STAT/BISR/Informatics/COD/LABELING/labled.csv'
EXCLUDE = ['the', 'are', 'of', 'as', 'is', 'and', 'or']
ALIASES = ['er', 'estrogen']

# Read in the pathology reports.
csv = pd.read_csv(PATHOLOGY_FILE, delimiter=',', encoding='ISO-8859-1')
hotspot_rows=csv.iloc[:,1]

st=LancasterStemmer()
#translator = str.maketrans('', '', string.punctuation)

voi_rows = []
all_words = []
for row in hotspot_rows:
    tok_row = sent_tokenize(row)
    voi_sentence = ""
    for sentence in tok_row:
        if re.search(".*\\b" + "\\b.*|.*\\b".join(ALIASES) + "\\b.*", sentence, flags = re.I):
            voi_sentence = sentence
            break
    voi_sentence = voi_sentence.lower()
    voi_sentence = re.sub(r'estrogen receptor', 'er', voi_sentence)
    voi_sentence = re.sub(r'estrogen', 'er', voi_sentence)
    voi_words = voi_sentence.split()
    #voi_words = [x.translate(translator) for x in voi_words]
    voi_words = [re.sub('[' + string.punctuation + ']', '', x) for x in voi_words]
    voi_words = list(filter(lambda x: x not in EXCLUDE, voi_words))
    voi_words = [st.stem(x) for x in voi_words]
    voi_rows.append(voi_words)
    all_words.extend(voi_words)

word_freq = {}
for word in all_words: 
    if word in word_freq.keys(): #.keys pick out each word
        word_freq[word] = word_freq[word] + 1
    else:
        word_freq[word] = 1 

top_words = sorted(word_freq, key=word_freq.__getitem__, reverse=True)[0:15]


# In[42]:

word_vectors = []

for row in voi_rows:
    row = list(filter(lambda x: x in top_words, row))
    if 'er' not in row:
        continue
    voi_loc = row.index('er')
    
    locs = [-10] * len(top_words)
    word_loc = {}
    
    row_enum = enumerate(row)
    for word in row_enum:
        if word[1] in word_loc.keys():
            if abs(voi_loc - word[0]) < abs(voi_loc - word_loc[word[1]]):
                word_loc[word[1]] = word[0]
        else:
            word_loc[word[1]] = word[0]
    
    for key, value in word_loc.items():
        top_loc = top_words.index(key)
        locs[top_loc] = value - voi_loc
    
    word_vectors.append(locs)


# In[43]:

word_vectors


# In[ ]:




# In[ ]:



