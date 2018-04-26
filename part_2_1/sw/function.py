import codecs
from bs4 import BeautifulSoup
import string
import nltk
import os.path
import math
import pandas as pd
nltk.download('punkt')

# Extract and clean songs
def extractSong(filename):
    punctuation_symbols = dict((ord(char), None) for char in string.punctuation)
    myFiles = codecs.open(filename,'r',encoding='utf8')
    soup = BeautifulSoup(myFiles.read(),"lxml")
    if 'br' not in [tag.name for tag in soup.find_all()]:
        return None
    title=os.path.basename(filename)
    text = str(soup.find("body")).replace("<br/>"," ").replace("</body>","").replace("<body>","")
    text_cleaned = [word.lower().translate(punctuation_symbols) for word in nltk.word_tokenize(text) if word.isalnum()]
    return {title:text_cleaned}

# Create the shingles
def shingling(dic):
    shingle_len=3; lyrics_shingle=set()
    lyrics=next(iter(dic.values()))
    for token in range(len(lyrics)-shingle_len+1):
        shingles=tuple(lyrics[token:token+shingle_len])
        lyrics_shingle.add(shingles)
    return lyrics_shingle

# replace the shingles with their respective ids
def rep_shingle(set_,dic):
    for key in dic:
        return {dic.get(elem) for elem in set_}

# Prime number
def is_prime(number):
	for j in range(2, int(math.sqrt(number)+1)):
		if (number % j) == 0: 
			return False
	return True

# Jaccard Similarity
def jaccard_similarity(lshmh_df, song_sh): # no set input number of publications as arguments
    js_values = []
    for song1,song2 in lshmh_df:
        intersection_cardinality = len(set.intersection(song_sh[song1],song_sh[song2]))
        union_cardinality = len(set.union(song_sh[song1],song_sh[song2]))
        js_values.append(intersection_cardinality/float(union_cardinality))
    lshmh_df = pd.concat([lshmh_df, pd.Series(js_values)], axis=1)
    lshmh_df.columns=['Near Duplicate Candidates','Jaccard Similarity']
    lshmh_df=lshmh_df.set_index('Near Duplicate Candidates')
    return lshmh_df
