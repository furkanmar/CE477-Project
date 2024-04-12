import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df = pd.read_csv('GameSalmy es.csv')
#minMaxScaler
scaler = MinMaxScaler()
#attributes for normalization
numeric_attributes = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
df[numeric_attributes] = scaler.fit_transform(df[numeric_attributes])

#normalized data added in new csv
df.to_csv('GameSales_Normalized.csv', index=False)

#normalized data
print("Normalized data:")
print(df[numeric_attributes].head())

#plot of normalized data
plt.figure(figsize=(10, 6))
for attribute in numeric_attributes:
    plt.plot(df[attribute], label=attribute)

plt.title('Normalized Sales Data')
plt.xlabel('Index')
plt.ylabel('Normalized Sales')
plt.legend()
plt.show()
