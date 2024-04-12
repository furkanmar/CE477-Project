from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('GameSales.csv')
scaler = StandardScaler()

#attributes selected
numeric_attributes = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']

#standardization
df[numeric_attributes] = scaler.fit_transform(df[numeric_attributes])

#new csv
df.to_csv('GameSales_Standardized.csv', index=False)

#standardized data
print("Standardized data:")
print(df[numeric_attributes].head())

#plot
plt.figure(figsize=(10, 6))
for attribute in numeric_attributes:
    plt.plot(df[attribute], label=attribute)

plt.title('Standardized Sales Data')
plt.xlabel('Index')
plt.ylabel('Standardized Sales')
plt.legend()
plt.show()
