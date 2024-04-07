import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('GameSales.csv')
df.rename(columns={'Global_Sales;': 'Global_Sales'}, inplace=True)
print(df.columns)



fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
# NA Sales vs Year
axes[0, 0].scatter(df['Year'], df['NA_Sales'], alpha=0.5)
axes[0, 0].set_title('North America Sales vs Year')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Sales in NA (Millions)')

# EU Sales vs Year
axes[0, 1].scatter(df['Year'], df['EU_Sales'], alpha=0.5, color='green')
axes[0, 1].set_title('Europe Sales vs Year')
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('Sales in EU (Millions)')

# JP Sales vs Year
axes[1, 0].scatter(df['Year'], df['JP_Sales'], alpha=0.5, color='red')
axes[1, 0].set_title('Japan Sales vs Year')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Sales in JP (Millions)')

# Other Sales vs Year
axes[1, 1].scatter(df['Year'], df['Other_Sales'], alpha=0.5, color='purple')
axes[1, 1].set_title('Other Sales vs Year')
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Sales in Other Regions (Millions)')

# Figure ve axes ayarları
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
df['Global_Sales'] = df['Global_Sales'].str.replace(';', '').astype(float)


# NA Sales vs Global Sales
axes[0, 0].scatter(df['Year'], df['NA_Sales'], alpha=0.5, color='blue', label='NA Sales')
axes[0, 0].scatter(df['Year'], df['Global_Sales'], alpha=0.1, color='black', label='Global Sales')
axes[0, 0].set_title('NA vs Global Sales by Year')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Sales (Millions)')
axes[0, 0].legend()

# EU Sales vs Global Sales
axes[0, 1].scatter(df['Year'], df['EU_Sales'], alpha=0.5, color='green', label='EU Sales')
axes[0, 1].scatter(df['Year'], df['Global_Sales'], alpha=0.1, color='black', label='Global Sales')
axes[0, 1].set_title('EU vs Global Sales by Year')
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('Sales (Millions)')
axes[0, 1].legend()

# JP Sales vs Global Sales
axes[1, 0].scatter(df['Year'], df['JP_Sales'], alpha=0.5, color='red', label='JP Sales')
axes[1, 0].scatter(df['Year'], df['Global_Sales'], alpha=0.1, color='black', label='Global Sales')
axes[1, 0].set_title('JP vs Global Sales by Year')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Sales (Millions)')
axes[1, 0].legend()

# Other Sales vs Global Sales
axes[1, 1].scatter(df['Year'], df['Other_Sales'], alpha=0.5, color='purple', label='Other Sales')
axes[1, 1].scatter(df['Year'], df['Global_Sales'], alpha=0.1, color='black', label='Global Sales')
axes[1, 1].set_title('Other vs Global Sales by Year')
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Sales (Millions)')
axes[1, 1].legend()




plt.tight_layout()
plt.show()