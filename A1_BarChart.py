import pandas as pd
import matplotlib.pyplot as plt

#Bar Chart task
df = pd.read_csv("GameSales.csv")

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
#na sales
axes[0, 0].bar(df.index, df["NA_Sales"])
axes[0, 0].set_title("North America Sales Bar Chart")
axes[0, 0].set_ylim([0, 15])  # y limit --> 15
#eu sales
axes[0, 1].bar(df.index, df["EU_Sales"])
axes[0, 1].set_title("Europe Sales Bar Chart")
axes[0, 1].set_ylim([0, 10])  # y limit --> 15
#japan sales
axes[1, 0].bar(df.index, df["JP_Sales"])
axes[1, 0].set_title("Japan Sales Bar Chart")
axes[1, 0].set_ylim([0, 8])  # y limit --> 15
#other sales
axes[1, 1].bar(df.index, df["Other_Sales"])
axes[1, 1].set_title("Other Sales Bar Chart")
axes[1, 1].set_ylim([0, 4])  # y limit --> 15

plt.tight_layout()
plt.show()

#I put a y-axis limit because on the other hand there will be very tiny plots, and we cannot observe the results.