# <a name="_35azdq8fxfe4"></a>**Video Game Sales Data Analysis**
## <a name="_52n2nhdrpu32"></a>**Project Overview**
This project focuses on a comprehensive analysis of global video game sales data. The dataset includes sales information for video games released on various platforms across different regions. The goal is to identify trends and patterns in sales data to help game developers and publishers make strategic decisions based on market insights.

The dataset, **GameSales.csv**, provides detailed insights into video game sales across regions and platforms. Using data processing techniques such as normalization, discretization, and outlier detection, this project aims to understand the dynamics of the video game market better and forecast future trends.
## <a name="_2cek50s4sncy"></a>**Project Goals**
The project is structured around six key objectives:

1. **Data Analysis**: Conducting a thorough analysis of video game sales data to identify regional trends and patterns in consumer preferences.
1. **Market Understanding**: Gaining insights into the video game market by analyzing sales data to determine the most popular platforms and genres.
1. **Historical Sales Evaluation**: Reviewing historical sales data to assess the performance of various publishers and titles over time.
1. **Regional Sales Comparison**: Comparing video game sales across different regions (North America, Europe, Japan, etc.) to evaluate market shares and consumer preferences.
1. **Sales Forecasting**: Using historical data to predict future sales trends and market dynamics.
1. **Strategic Decision Making**: Providing game publishers and developers with actionable insights to improve game development, marketing, and distribution strategies.
## <a name="_8kj23n2h6bzr"></a>**Dataset Description**
The dataset contains sales data for over 16,000 video games. Below are the key attributes included in the dataset:

- **Rank**: The rank of the game based on global sales.
- **Name**: The title of the video game.
- **Platform**: The platform on which the game was released (e.g., PC, PS4).
- **Year**: The year the game was released.
- **Genre**: The genre of the game (e.g., Action, Puzzle).
- **Publisher**: The publisher of the game.
- **North America Sales**: Sales in North America (in millions).
- **Europe Sales**: Sales in Europe (in millions).
- **Japan Sales**: Sales in Japan (in millions).
- **Other Sales**: Sales in the rest of the world (in millions).
- **Global Sales**: Total worldwide sales (in millions).
### <a name="_3rqzayffljhj"></a>**Dataset Details**
- **File format**: CSV
- **Source**: Scraped from VGChartz using BeautifulSoup in Python.
- **Size**: 16,598 records with 11 attributes.
- **Missing Data**: No missing values in the dataset.
## <a name="_7qyto45n2vzi"></a>**Methodology**
The following data processing and analysis techniques were used throughout the project:

1. **Preprocessing**:
   1. Handling missing values and identifying outliers.
   1. Normalizing and discretizing data for more effective analysis.
1. **Visualizations**:
   1. Using box plots, scatter plots, and bar charts to illustrate trends and data distributions.
   1. Correlation matrices to analyze relationships between different sales attributes and global sales.
1. **Predictive Analysis**:
   1. Classification algorithms (Decision Trees, k-Nearest Neighbors, Random Forest) were applied to classify games based on platform, publisher, and genre.
   1. Regression analysis was performed to predict future sales values.
## <a name="_ynlh07t94z8"></a>**Data Analysis**
### <a name="_m67gxuc8f94q"></a>**Correlation Analysis**
A correlation matrix was used to analyze relationships between sales in different regions. Strong positive correlations were found between North America, Europe, Japan sales, and global sales, indicating that high sales in one region typically correspond to high global sales. The release year showed weak correlation with sales, suggesting that sales are not highly dependent on the year of release.
### <a name="_7pkq8jlrtmuu"></a>**Boxplot and Outlier Detection**
Boxplots revealed that the majority of games achieve low sales, while a small number of games dominate the market with extremely high sales. These outliers represent a small number of highly successful games compared to the overall market.
### <a name="_m7b31k4y5nh6"></a>**Classification Models**
Classification models such as Decision Trees, k-Nearest Neighbors, and Random Forest were used. The **Random Forest** model provided the highest accuracy in predicting platforms, making it the most reliable for this dataset.
### <a name="_k4omcrhhbs7"></a>**Regression Analysis**
Linear regression models were applied to predict global sales. The model performed well, with accurate predictions for most sales data. The RMSE and MAPE values confirmed the model's reliability for future sales forecasting.
## <a name="_u7xvzsq4nvz9"></a>**Technologies Used**
This project utilizes the following technologies:

- **Python**: For data preprocessing and analysis.
- **Pandas & NumPy**: For data manipulation and analysis.
- **Matplotlib & Seaborn**: For data visualization.
- **scikit-learn**: For implementing machine learning models.
## <a name="_xeke5ytxapq"></a>**Key Findings**
- **Regional Trends**: North America and Europe generally show higher sales figures than Japan and other regions.
- **Outliers**: A small percentage of games account for the majority of high sales, indicating a competitive market.
- **Sales Forecasting**: The regression models provided highly accurate sales predictions, supporting strategic decision-making for game developers and publishers.

