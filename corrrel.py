
import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd

data=load_breast_cancer()["data"]

column=load_breast_cancer()["feature_names"]


df=pd.DataFrame(columns=column,data=data)




#df=pd.DataFrame({"A":[1,2,3,4,5,6,7],"B":[2,4,6,8,10,12,14],"C":[3,6,8,9,1,2,7],"D":[3,6,9,12,15,18,21]})

#print(df)
def column_reduction(data=df,corr=0.89):

    j=df.corr().abs()
    np.fill_diagonal(j.values,0)

    p=[]
    indices, columns=np.where(j>.89)
    t=pd.DataFrame(list(np.where((j>.89)))).T



    t.columns=["A","B"]

    q=set(df.columns[list(set(np.append(t["A"],t["B"],axis=0)))])


    d=t.groupby("A")
    z=d.agg(A1=("A",np.max),B1=("B",np.max))

    h=len(z.index)
    w=set()
    for i in range(h):
        e=(z.iloc[i,0],z.iloc[i,1])
        
        w.add(max(e))

    data_column=set(df.columns)-q
    data_column.update(set(df.columns[list(w)]))
    new_columns=list(data_column)
    new_columns.sort()
    modify_data=df[new_columns]

    old, new=len(df.columns) , len(modify_data.columns)
    print(f"colums reduce from {old}, columns to {new} columns")
    return (modify_data,new_columns)

print(column_reduction())