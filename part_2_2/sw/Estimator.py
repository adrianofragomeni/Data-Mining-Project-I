import ast
import numpy as np

# Compute the length of a given set by using an approximation of the JS
def find_length_set(str_sketch,len_uni=1123581321):
    lst_sketch=ast.literal_eval(str_sketch)
    count_min=lst_sketch.count(min(lst_sketch))
    approx_js=count_min/len(lst_sketch)
    return int(len_uni*approx_js)

# Compute the length of a Union of sets by calling the function above
def find_union_sketch(union_set,df):
    lst_union=list(ast.literal_eval(union_set))
    df=df.loc[lst_union,:]['Min_Hash_Sketch']
    t_matrix=np.transpose(list(map(ast.literal_eval,df.values)))
    union_sketch=[min(val) for val in t_matrix]
    return find_length_set(str(union_sketch))
    