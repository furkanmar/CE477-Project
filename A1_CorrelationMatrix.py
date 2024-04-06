import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("GameSales.csv")
correlation_matrix = data.corr()  #it calculate the correlation matrix from the data

sns.heatmap(correlation_matrix, annot=True,linewidths=.5)
#annot=true -> fill the middle of the box with its value
#linewidth define the "çizgi kalınlığı" between cels
# if fmt=".2f" added to the funtion it format the values with 2 after comma value


plt.show()
