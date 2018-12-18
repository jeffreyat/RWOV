# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:19:04 2017

@author: jthompson21
"""

import re  
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import sent_tokenize
import string
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def which_min(x):
    """Return index in list of minimum value.
    """
    return x.index(min(x))

def get_shortest_alias(aliases):
    """Return the shortest alias from a list.
    """
    return(aliases[which_min([len(x) for x in aliases])])

def sub_short_alias(aliases, target):
    """Substitute shortest alias for all occurences of longer aliases.
    """
    min_alias = which_min([len(x) for x in aliases])
    other_aliases = [x for x in range(len(aliases)) if x != min_alias]
    for i in range(len(other_aliases)):
        target = re.sub(aliases[other_aliases[i]], aliases[min_alias], target)
    return(target)

def sub_percent(voi, target):
    """Substitute a string for percentages.
    """
    target_words = target.split()
    seen_voi = False
    for i in range(len(target_words)):
        if target_words[i] == voi:
            seen_voi = True
        if not seen_voi and re.search(r'\d+%,', target_words[i]):
            target_words[i] = 'comma_before'
        pct_search = re.search(r'\d+%', target_words[i])
        if pct_search:
            if float(pct_search.group().strip('%')) > 1:
                target_words[i] = "pospct"
            else:
                target_words[i] = "negpct"
    return(" ".join(target_words))

def sub_symbols(voi, target):
    """Substitute strings for some symbols. Currently a little hackey. 
    """
    target_words = target.split()
    for i in range(len(target_words)):
        plus_search = re.search(r'\+', target_words[i])
        if plus_search:
            target_words[i] = "plus"
        comma_search = re.search(r',', target_words[i])
        if comma_search:
            target_words[i] = "comma"
        slash_search = re.search(r'/', target_words[i])
        if slash_search:
            target_words[i] = "slash"
    return(" ".join(target_words))

def get_voi_rows(voi, hotspot_rows, aliases, exclude):
    """This function is a little misnamed at this point, due to
    refactoring. It returns rows containing the voi that have been
    stemmed, as well as all words encountered.
    """
    
    # Lists that contain rows in which VOI occurs and of all words.
    voi_rows = []
    all_words = []
    
    # Create stemmer and translator objects.
    st=LancasterStemmer()

    # Get stemmed rows in containing VOI and collect all stemmed words.
    for row in hotspot_rows:
        tok_row = sent_tokenize(row)
        voi_sentence = ""
        for sentence in tok_row:
            if re.search(".*\\b" + "\\b.*|.*\\b".join(aliases) + "\\b.*", 
                         sentence, flags = re.I):
                voi_sentence = sentence
                break
        voi_sentence = voi_sentence.lower()
        voi_sentence = sub_short_alias(aliases, voi_sentence)
        voi_sentence = sub_percent(voi, voi_sentence)
        voi_sentence = sub_symbols(voi, voi_sentence)
        voi_words = voi_sentence.split()
        # Patch for Python3
        #voi_words = [x.translate(translator) for x in voi_words]
        voi_words = [str(x).translate(None, string.punctuation) for x in voi_words]
        #voi_words = [str(x).translate(None, string.punctuation.replace("+,-", ",")) for x in voi_words]
        voi_words = list(filter(lambda x: x not in exclude, voi_words))
        voi_words = [str(st.stem(x)) for x in voi_words]
        voi_rows.append(voi_words)
        all_words.extend(voi_words)

    return((voi_rows, all_words))

def get_top_words(all_words, num_top_words):
    # Count the frequency of words occuring in rows with VOI.
    word_freq = {}
    for word in all_words: 
        if word in word_freq.keys(): #.keys pick out each word
            word_freq[word] = word_freq[word] + 1
        else:
            word_freq[word] = 1 

    # Get a list of top-occuring words.
    top_words = sorted(word_freq, key=word_freq.__getitem__, reverse=True)[0:num_top_words]
    
    return(top_words)

def get_word_vectors(voi, voi_rows, top_words, default_rel_loc):
    """Return a vectorized form of the data.
    """
    # A list of vectors of the relative location of top words to the VOI.
    word_vectors = []
    
    # Calculate list of vectors where each vector corresponds
    # to relative location of nearest top word to the VOI (if it occurs).
    for row in voi_rows:
        row = list(filter(lambda x: x in top_words, row))
        voi_loc = row.index(voi)
        
        locs = [default_rel_loc] * len(top_words)
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
            if locs[top_loc] != 0:
                locs[top_loc] = 1./locs[top_loc]
        
        word_vectors.append(locs)
    return(word_vectors)

def get_weighted_vectors(word_vectors, y, top_words, 
                         voi, voi_rows, rel_loc,
                         nearness=5,
                         new_top_num=15):
    """Return a modified vectorized form of the data which remove words
    that do not differentiate between classes very well.
    """
    scaler = StandardScaler()

    scaler.fit(word_vectors)
    result_scaled = scaler.transform(word_vectors)
    result_np = np.array(result_scaled)

    distances = euclidean_distances(result_np, result_np)

    # Experimental weighting by RELIEF for selecting only the best words for vectorization.
    weights = [0] * result_np.shape[1]
    
    for m in range(1):
        for i in range(result_np.shape[0]):
            row_sort = sorted(range(len(distances[i])), key=lambda k: distances[i][k])
            row_class = y[i]
            
            shared_class = [x for x in row_sort if row_class == y[x]]
            diff_class = [x for x in row_sort if row_class != y[x]]
            
            weights = weights - (result_np[i] - np.mean([result_np[j] for j in shared_class[1:nearness]], axis=0))**2 + (result_np[i] - np.mean([result_np[j] for j in diff_class[1:nearness]], axis=0))**2
    
    sorted_weights = sorted(range(len(weights)), key=lambda k: weights[k], reverse=True)
    new_top_words = [top_words[i] for i in range(len(top_words)) if i in sorted_weights[0:(new_top_num+1)] or top_words[i] == voi]
    
    # Get stemmed rows in containing VOI and collect all stemmed words.
    word_vectors = get_word_vectors(voi, voi_rows, new_top_words, rel_loc)

    return(word_vectors)

def get_new_top_words(word_vectors, y, top_words, 
                         voi, voi_rows, rel_loc,
                         nearness=5,
                         new_top_num=15):
    """Return a modified vectorized form of the data which remove words
    that do not differentiate between classes very well.
    """
    scaler = StandardScaler()

    scaler.fit(word_vectors)
    result_scaled = scaler.transform(word_vectors)
    result_np = np.array(result_scaled)

    distances = euclidean_distances(result_np, result_np)

    # Experimental weighting by RELIEF for selecting only the best words for vectorization.
    weights = [0] * result_np.shape[1]
    
    for m in range(1):
        for i in range(result_np.shape[0]):
            row_sort = sorted(range(len(distances[i])), key=lambda k: distances[i][k])
            row_class = y[i]
            
            shared_class = [x for x in row_sort if row_class == y[x]]
            diff_class = [x for x in row_sort if row_class != y[x]]
            
            weights = weights - (result_np[i] - np.mean([result_np[j] for j in shared_class[1:nearness]], axis=0))**2 + (result_np[i] - np.mean([result_np[j] for j in diff_class[1:nearness]], axis=0))**2
    
    sorted_weights = sorted(range(len(weights)), key=lambda k: weights[k], reverse=True)
    new_top_words = [top_words[i] for i in range(len(top_words)) if i in sorted_weights[0:(new_top_num+1)] or top_words[i] == voi]
    
    return(new_top_words)