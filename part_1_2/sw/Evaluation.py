import pandas as pd
import numpy as np

class Eval_Engine():
        
# Group the SEs and the Ground Truth by each query id and randomize the order of the results
    def __init__(self,G_truth,SE1,SE2,SE3):
        self.G_truth=G_truth.groupby('Query_id')
        self.SE1=application(SE1.groupby('Query_ID'))
        self.SE2=application(SE2.groupby('Query_ID'))
        self.SE3=application(SE3.groupby('Query_ID'))

# Precision at K
    def Precision_at_K(self,group,K):
            count=0
            for val in group.iloc[:K,]['Doc_ID']:
                if val in self.G_truth.get_group(group['Query_ID'].values[0])['Relevant_Doc_id'].values:
                    count+=1
            return count/K 
    
# Take the same Query for GT and SE
    def clean_data(self,SE):
        df=SE.filter(lambda x: x['Query_ID'].values[0] in self.G_truth.groups.keys())
        return df.groupby('Query_ID')
       

# Apply the method and return a dataframe
    def apply_evaluation(self,method):
        Search_Engine=[self.SE1,self.SE2,self.SE3]
        if method=='P@k':
            return pd.DataFrame([self.clean_data(SE).apply(self.Precision_at_K,4).mean() for SE in Search_Engine],
                     columns=["Mean(P@4)"],
                     index=['SE_1','SE_2','SE_3'])
            
# Randomize the order of the results    
def application(SE):
    result_app=pd.DataFrame()
    for ind,df in SE:
        result_app=result_app.append(df.head(4).transform(np.random.permutation))
    return result_app.groupby('Query_ID')
            
            
