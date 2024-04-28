import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("GameSales.csv")
#Box plot task
def boxplot(df):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 5))
    # na sales

    axes[0, 0].boxplot(df["NA_Sales"])
    axes[0, 0].set_title("North America Sales Box Plot")
    # eu sales
    axes[0, 1].boxplot(df["EU_Sales"])
    axes[0, 1].set_title("Europe Sales Box Plot")
    # japan sales
    axes[1, 0].boxplot(df["JP_Sales"])
    axes[1, 0].set_title("Japan Sales Box Plot")
    # other sales
    axes[1, 1].boxplot(df["Other_Sales"])
    axes[1, 1].set_title("Other Sales Box Plot")
    # show the grap
    plt.tight_layout()
    plt.show()

boxplot(df)