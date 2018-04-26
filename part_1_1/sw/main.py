import pandas as pd
import Evaluation as Ev


# Transform the datasets into Pandas Dataframes
GT=pd.read_csv('part_1_1__Ground_Truth.tsv', sep='\t',index_col=None)
SE_1=pd.read_csv('part_1_1__Results_SE_1.tsv', sep='\t',index_col='Rank')
SE_2=pd.read_csv('part_1_1__Results_SE_2.tsv', sep='\t',index_col='Rank')
SE_3=pd.read_csv('part_1_1__Results_SE_3.tsv', sep='\t',index_col='Rank')

# Create an Evaluation object
Evaluate=Ev.Eval_Engine(GT,SE_1,SE_2,SE_3)

# Precision at K
Evaluate.apply_evaluation('P@k').to_csv('p_at_k.csv',index_label='Search Engine')

# Mean Reciprocal Rank
Evaluate.apply_evaluation('MRR').to_csv('MRR.csv',index_label='Search Engine')

# R-precision
Evaluate.apply_evaluation('Rprecision').to_csv('R_precision.csv',index_label='Search Engine')

# Normalised_DCG
Evaluate.apply_evaluation('NDCG').to_csv("nDCG.csv", index_label="Search Engine")
