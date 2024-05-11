import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from sklearn import tree
# Load the dataset
file_path = 'GameSales_dropped.csv'
game_sales_data = pd.read_csv(file_path)
#game_sales_data.rename(columns={'Global_Sales;': 'Global_Sales'}, inplace=True)
#game_sales_data['Global_Sales'] = game_sales_data['Global_Sales'].str.replace(';', '').astype(float)

# Choosing 'Global_Sales' as the target variable
X = game_sales_data.drop(['Global_Sales', 'Name', 'Publisher', 'Rank'], axis=1)  # Dropping non-numeric and target columns
y = game_sales_data['Global_Sales']

# Encoding categorical variables
X = pd.get_dummies(X, drop_first=True)

# Handling missing values by dropping them
X.dropna(inplace=True)
y = y[X.index]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Making predictions with the Linear Regression model
y_pred_linear = linear_model.predict(X_test)

# Calculating RMSE and MAPE for the Linear Regression
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
mape_linear = mean_absolute_percentage_error(y_test, y_pred_linear)

# Creating and training the Decision Tree Regressor
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# Making predictions with the Decision Tree Regressor
y_pred_tree = tree_model.predict(X_test)

# Calculating RMSE and MAPE for the Decision Tree Regressor
rmse_tree = np.sqrt(mean_squared_error(y_test, y_pred_tree))
mape_tree = mean_absolute_percentage_error(y_test, y_pred_tree)

rmse_linear, mape_linear, rmse_tree, mape_tree
print("Linear Regression RMSE:", rmse_linear)
print("Linear Regression MAPE:", mape_linear)
print("Decision Tree RMSE:", rmse_tree)
print("Decision Tree MAPE:", mape_tree)

# Scatter plot for Linear Regression model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_linear, color='blue', alpha=0.5, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4, label='Ideal Line')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression: Actual vs Predicted Values')
plt.legend()
plt.grid(True)
#plt.savefig('Linear Regression: Actual vs Predicted Values.pdf')
#plt.close()
plt.show()

# Residual plot for Linear Regression
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred_linear
plt.scatter(y_test, residuals, color='red', alpha=0.5)
plt.hlines(y=0, xmin=y_test.min(), xmax=y_test.max(), linestyles='dashed')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.title('Linear Regression: Residuals Plot')
plt.grid(True)
#plt.savefig('Linear Regression: Residuals Plot.pdf')
#plt.close()
plt.show()

# Decision Tree visualization
plt.figure(figsize=(20, 10))
tree.plot_tree(tree_model, filled=True, feature_names=X.columns, max_depth=3, fontsize=10) #fpr maximum 3 features
plt.title('Decision Tree Regressor Visualization (first 3 levels)')
plt.savefig('Decision Tree Regressor Visualization (first 3 levels).pdf')
plt.close()
#plt.show()

plt.figure(figsize=(30, 20))
tree.plot_tree(tree_model, filled=True, feature_names=X.columns,max_depth=20, fontsize=10)
plt.title('Complete Decision Tree Regressor Visualization')
plt.savefig('Complete Decision Tree Regressor Visualization.pdf')
plt.close()
#plt.show()



# Feature importance for Decision Tree Regressor
importances = tree_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title('Feature Importances by Decision Tree Regressor')
plt.bar(range(X_train.shape[1]), importances[indices], color="green", align="center")
plt.xticks(range(X_train.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.savefig('Feature Importances by Decision Tree Regressor.pdf')
plt.close()
#plt.show()