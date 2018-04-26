import glob
import function as f
import os.path
import functools
import random
from operator import is_not
import pandas as pd

# Extract the songs and store them in a list
path = 'lyrics_collection/lyrics_collection__CONVERTED/'
List_song=list(map(f.extractSong,glob.glob(os.path.join(path, '*.html'))))

# Filter the list and remove the None values
List_song=list(filter(functools.partial(is_not, None), List_song))

# Create the set of shingles of all songs
shingles=list(map(f.shingling, List_song))

# Universe of shingles
Union = set.union(*shingles)

# Define unique shingles' ids
shingles_id={tot_shingles:idx for idx,tot_shingles in enumerate(Union)}

# Replace each shingle with its respective id
unique_shingles=list(map(functools.partial(f.rep_shingle,dic=shingles_id),shingles))

# Re-elaboration of the given code of the Hash functions
num_hash_functions = 300
upper_bound_on_number_of_distinct_terms  = len(Union)

# generate the hash functions and save the results on a tsv file
with open("300_hash_functions_file.tsv", "w") as hash_functions:
    hash_functions.write("a	 b	 p	 n\n")
    for hash_function_id in range(num_hash_functions):
        a = random.randint(1, upper_bound_on_number_of_distinct_terms-1)
        b = random.randint(0, upper_bound_on_number_of_distinct_terms-1)
        p = random.randint(upper_bound_on_number_of_distinct_terms, 10*upper_bound_on_number_of_distinct_terms)
        while f.is_prime(p) == False:
            p = random.randint(upper_bound_on_number_of_distinct_terms, 10*upper_bound_on_number_of_distinct_terms)
        hash_functions.write(str(a) + "\t" + str(b) + "\t" + str(p) + "\t" + str(upper_bound_on_number_of_distinct_terms) + "\n")

# Create a file to use as input when computing the LSH and mean-Hashing using the Java command line and store them in a dictionary
song_shingles={}
with open("all_lyrics.tsv", "w") as lyrics_shingles:
    lyrics_shingles.write("set_id	set_as_list_of_elements_id\n")
    for ind in range(len(List_song)):
        song=next(iter(List_song[ind].values()))
        title=list(List_song[ind].keys())[list(List_song[ind].values()).index(song)]
        song_shingles[title]=unique_shingles[ind]
        lyrics_shingles.write(title + "\t" + str(list(unique_shingles[ind])) + "\n")

# Near-duplicate candidates
LSH_minhash=pd.read_csv('output_data/near_duplicate_candidates.tsv', sep='\t',index_col=None,usecols=['name_set_1','name_set_2']).apply(tuple, axis=1)
lsh_mh=f.jaccard_similarity(LSH_minhash,song_shingles)

# False positives
falsePositives=lsh_mh.where(lsh_mh<0.85).dropna()
falsePositives.columns=['Jaccard Similarity']
falsePositives.to_csv("FalsePositives.csv",index_label='False Positives')

# True near duplicates
nearDuplicates=lsh_mh.where(lsh_mh>=0.85).dropna()
nearDuplicates.columns=['Jaccard Similarity']
nearDuplicates.to_csv("NearDuplicates.csv",index_label='Real Near Duplicates')



 

