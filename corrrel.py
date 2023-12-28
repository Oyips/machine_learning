import numpy as np
import pandas as pd


""" The column_reduction function takes in a dataset with it features
as column name in the form of a
dataframe and a threshold corr. As an example if the correlation
pairwise correlation for Column A, B, C is atleast the value of
the threshold it returns only one column among A, B, C."""


# data=load_breast_cancer()["data"]
# column=load_breast_cancer()["feature_names"]
# df=pd.DataFrame(columns=column,data=data)

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

    old, new=len(df.columns) , len(modify_data.columns)
    print(f"colums reduce from {old}, columns to {new} columns")
    return new_columns

