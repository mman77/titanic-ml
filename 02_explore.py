import pandas as pd
df = pd.read_csv("titanic.csv")
print(df.shape)
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.groupby("sex")["survived"].mean())
print(df.groupby("pclass")["survived"].mean())