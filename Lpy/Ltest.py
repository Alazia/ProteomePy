import pandas as pd
import numpy as np
data=pd.read_csv('../test/xx_4758prot_21sample.csv',header=0,index_col=0)
print(data)
rep=[[1,2,3], [4,5,8]]
columns=sum(rep, [])
print(columns)
df=data.T
pd.set_option('mode.chained_assignment', None)
df['rep']=df.index
for i in range(len(rep)):
    df['rep'][rep[i]]='rep_'+str(i)
dff=df.iloc[columns,:]
dff.index=dff['rep'].values
dff=dff.drop('rep',axis=1)

print(dff)
