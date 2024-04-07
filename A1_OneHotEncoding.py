import pandas as pd

df=pd.read_csv("GameSales.csv")  #data frame set
one_hot_encoded = pd.get_dummies(df['Platform'])  #apply one hot coding to Platform
df_encoded = pd.concat([df, one_hot_encoded], axis=1) #concatenate OHC df and original df
df_encoded.drop('Platform', axis=1, inplace=True)#drop the original genre attribute

print(df_encoded)

