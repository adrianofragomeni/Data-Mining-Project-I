import pandas as pd
import Evaluation as Ev

# Transform the datasets into Pandas Dataframes
GT_2=pd.read_csv('part_1_2__Ground_Truth.tsv', sep='\t',index_col=None)
SE_1_2=pd.read_csv('part_1_2__Results_SE_1.tsv', sep='\t',index_col='Rank')
SE_2_2=pd.read_csv('part_1_2__Results_SE_2.tsv', sep='\t',index_col='Rank')
SE_3_2=pd.read_csv('part_1_2__Results_SE_3.tsv', sep='\t',index_col='Rank')

# Create an App object
App=Ev.Eval_Engine(GT_2,SE_1_2,SE_2_2,SE_3_2)

# Precision at K
App.apply_evaluation('P@k').to_csv('p_at_k2.csv',index_label='Search Engine')

