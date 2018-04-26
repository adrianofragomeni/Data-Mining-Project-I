import pandas as pd
import Estimator as E

# Transform the file into a pandas dataframe
minhash_set=pd.read_csv('HW_1_part_2_2_dataset__min_hash_sketches.tsv', sep='\t',
                        index_col='Min_Hash_Sketch_INTEGER_Id',usecols=['Min_Hash_Sketch_INTEGER_Id','Min_Hash_Sketch'])

# Apply the function to estimate the length of the set
minhash_set['Estimated_original_set_size']=minhash_set.apply(lambda row: E.find_length_set(row.Min_Hash_Sketch),axis=1)

# Save into a csv file
minhash_set.to_csv('Output_HW_1_part_2_2_a.csv',index_label='Min_Hash_Sketch_INTEGER_Id',columns=['Estimated_original_set_size'])


# Transform the file into a pandas dataframe
set_union=pd.read_csv('HW_1_part_2_2_dataset__SETS_IDS_for_UNION.tsv', sep='\t',
                        index_col='Union_Set_id')

# Apply the function to estimate the length of the union set
set_union['Estimated_union_size']=set_union.apply(lambda row:E.find_union_sketch(row.set_of_sets_ids,minhash_set),axis=1)

# Save into a csv file
set_union.to_csv('Output_HW_1_part_2_2_b.csv',index_label='Union_Set_id')




