import pandas as pd
from numpy import inf
import matplotlib.pyplot as plt


df = pd.read_csv('GameSales.csv')
df.rename(columns={'Global_Sales;': 'Global_Sales'}, inplace=True)
print(df.columns)
df['Global_Sales'] = df['Global_Sales'].str.replace(';', '').astype(float)

# IQR hesaplama
Q1 = df['Global_Sales'].quantile(0.25)
Q3 = df['Global_Sales'].quantile(0.75)
IQR = Q3 - Q1




non_outliers_iqr = df[(df['Global_Sales'] >= (Q1 - 1.5 * IQR)) & (df['Global_Sales'] <= (Q3 + 1.5 * IQR))]


print("\nNon-Outliers using IQR Method:\n", non_outliers_iqr[['Global_Sales']].head())

outliers_iqr = df[(df['Global_Sales'] < (Q1 - 1.5 * IQR)) | (df['Global_Sales'] > (Q3 + 1.5 * IQR))]


print("Outliers using IQR Method:\n", outliers_iqr[['Global_Sales']])

plt.figure(figsize=(10, 6))
plt.boxplot(df['Global_Sales'].dropna(), vert=False)  # NaN değerlerini çıkar ve yatay boxplot çiz
plt.title('Boxplot of Global Sales')
plt.xlabel('Global Sales (Millions)')
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(outliers_iqr.index, outliers_iqr['Global_Sales'], color='red', label='Outliers')
plt.scatter(df.index, df['Global_Sales'], color='blue', alpha=0.5, label='All Data')
plt.legend()
plt.title('Scatter Plot of Global Sales Highlighting Outliers')
plt.xlabel('Index')
plt.ylabel('Global Sales (Millions)')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(non_outliers_iqr.index, non_outliers_iqr['Global_Sales'], color='red', label='Non-Outliers')
plt.scatter(df.index, df['Global_Sales'], color='blue', alpha=0.5, label='All Data')
plt.legend()
plt.title('Scatter Plot of Global Sales Highlighting Non-Outliers')
plt.xlabel('Index')
plt.ylabel('Global Sales (Millions)')
plt.show()


sales_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # 4 subplot için 2x2 grid

for i, col in enumerate(sales_columns):
    ax = axes.flatten()[i]

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1


    outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]



    ax.scatter(df.index, df[col], color='blue', alpha=0.5, label='All Data')

    ax.scatter(outliers.index, outliers[col], color='red', label='Outliers')
    ax.set_title(f'{col} Outliers')
    ax.set_xlabel('Index')
    ax.set_ylabel(f'{col} (Millions)')
    ax.legend()

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, col in enumerate(sales_columns):
    ax = axes.flatten()[i]

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1


    non_outliers = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]

    # Scatter plot
    ax.scatter(non_outliers.index, non_outliers[col], color='blue', alpha=0.5, label='Non-Outliers')
    ax.set_title(f'{col} Non-Outliers')
    ax.set_xlabel('Index')
    ax.set_ylabel(f'{col} (Millions)')
    ax.legend()

plt.tight_layout()
plt.show()