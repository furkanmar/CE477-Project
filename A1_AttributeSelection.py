import pandas as pd
from scipy.stats import spearmanr

df = pd.read_csv("GameSales.csv")

#if it is NaN then drop it.
df = df.dropna(subset=['Year', 'Global_Sales'])

#is it numeric? ensure.
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

#attributes
global_sales = df['Global_Sales']
year = df['Year']

#spearman's rank correlation
correlation, p_value = spearmanr(year, global_sales)

#results
print(f"Spearman's rank correlation coefficient: {correlation:.3f}")
print(f"P-value: {p_value:.3f}")

#interpretation
if p_value < 0.05:
    print("The correlation is statistically significant.")
else:
    print("The correlation is not statistically significant.")
