import pandas as pd
from numpy import inf
import matplotlib.pyplot as plt


df = pd.read_csv('GameSales.csv')
df.rename(columns={'Global_Sales;': 'Global_Sales'}, inplace=True)
print(df.columns)
df['Global_Sales'] = df['Global_Sales'].str.replace(';', '').astype(float)
global_sales_mean = df['Global_Sales'].mean()

print("Global Sales Ortalaması:", global_sales_mean)


mean = global_sales_mean
delta = mean * 0.5

# Bins listesini tanımla
bins = [mean - delta, mean, mean + delta, inf]

labels = ['Low', 'Middle', 'High']
df['Sales_Category'] = pd.cut(df['Global_Sales'], bins=bins, labels=labels)


print(df[['Global_Sales', 'Sales_Category']].head())

category_counts = df['Sales_Category'].value_counts()


category_counts.plot(kind='bar')
plt.title('Sales Categories Distribution')
plt.xlabel('Sales Category')
plt.ylabel('Number of Games')
plt.xticks(rotation=0)
plt.show()

df['Sales_Frequency_Category'] = pd.qcut(df['Global_Sales'], 4, labels=['1', '2', '3', '4'])


print(df[['Global_Sales', 'Sales_Frequency_Category']].head())
frequency_category_counts = df['Sales_Frequency_Category'].value_counts().sort_index()


frequency_category_counts.plot(kind='bar')
plt.title('Sales Frequency Categories Distribution')
plt.xlabel('Sales Frequency Category')
plt.ylabel('Number of Games')
plt.xticks(rotation=0)
plt.show()