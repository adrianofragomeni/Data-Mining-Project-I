import pandas as pd
import numpy as np

class Eval_Engine():
    
# Group the SEs and the Ground Truth by each query id
    def __init__(self,G_truth,SE1,SE2,SE3):
        self.G_truth=G_truth.groupby('Query_id')
        self.SE1=SE1.groupby('Query_ID')
        self.SE2=SE2.groupby('Query_ID')
        self.SE3=SE3.groupby('Query_ID')

# Precision at K
    def Precision_at_K(self,group,K):
        count=0
        for val in group.iloc[0:K,]['Doc_ID']:
            if val in self.G_truth.get_group(group['Query_ID'].values[0])['Relevant_Doc_id'].values:
                count+=1
        return count/K 

# Mean Reciprocal Rank
    def Mean_Reciprocal_Rank(self,group):
        for val in group['Doc_ID']:
            if val in self.G_truth.get_group(group['Query_ID'].values[0])['Relevant_Doc_id'].values:
                position=group[group['Doc_ID']==val].index[0]
                return 1/position

# R-precision
    def R_prec(self,group):
        k=len(self.G_truth.get_group(group['Query_ID'].values[0]).index)
        count=0
        for val in group.iloc[0:k,]['Doc_ID']:
            if val in self.G_truth.get_group(group['Query_ID'].values[0])['Relevant_Doc_id'].values:
                count+=1 
        return count/k
    
    
# Normalized Discounted Cumulative Gain
    def Normalised_DCG(self,group, K):
            relevance=np.array([1 if result in self.G_truth.get_group(group["Query_ID"].values[0])["Relevant_Doc_id"].values else 0 for result in group.iloc[:K,:]["Doc_ID"]][:K])
            DCG=relevance[0]+np.sum(relevance[1:]/np.log2(np.arange(2,relevance.size+1)))
            IDCG=1+np.sum(np.ones(len(relevance))[1:]/np.log2(np.arange(2,relevance.size+1)))
            return DCG/IDCG


# Apply the selected method and return a dataframe 
    def apply_evaluation(self,method):
        
        Search_Engine=[self.SE1,self.SE2,self.SE3]  
        if method=='P@k':
            return pd.DataFrame([[SE.apply(self.Precision_at_K,k).mean() for k in [1,3,5,10]] for SE in Search_Engine],
                     columns=["Mean(P@1)", "Mean(P@3)", "Mean(P@5)", "Mean(P@10)"],
                     index=['SE_1','SE_2','SE_3'])
            
        elif method=='Rprecision':
            return pd.DataFrame([np.delete(SE.apply(self.R_prec).describe().values,[0,2]) for SE in Search_Engine],
                     columns=["Mean (R-Precision_Distribution)","min(R-Precision_Distribution)","1°_quartile (R-Precision_Distribution)",
                              "MEDIAN(R-Precision_Distribution)","3°_quartile (R-Precision_Distribution)","MAX(R-Precision_Distribution)"],
                     index=['SE_1','SE_2','SE_3'])
            
        elif method=='NDCG':
            return pd.DataFrame([[SE.apply(self.Normalised_DCG, k).mean() for k in [1,3,5,10]] for SE in Search_Engine],
                                   columns=["Mean(nDCG@1)", "Mean(nDCG@3)", "Mean(nDCG@5)", "Mean(nDCG@10)"], index=["SE_1", "SE_2", "SE_3"])
        
        elif method=='MRR':
            return pd.DataFrame([SE.apply(self.Mean_Reciprocal_Rank).sum()/len(self.G_truth) for SE in Search_Engine],
                     columns=["MRR"],
                     index=['SE_1','SE_2','SE_3'])
            
