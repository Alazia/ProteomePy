import pandas as pd
p_value=0.05
df=pd.read_csv('../test/down.csv',nrows=5)
print(df)
m=len(df['p_value'])
df['p_rank']=df['p_value'].rank(ascending=True)
df['p_adjust_value']=df['p_value']*(m/df['p_rank'])
df['p_k']=p_value*df['p_rank']/m
min_rank=min(df[df['p_adjust_value']>df['p_k']]['p_rank'])
df[df['p_rank']>=min_rank]['p_adjust_value']=df['p_value']
print(df)