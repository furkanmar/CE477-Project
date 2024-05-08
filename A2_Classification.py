import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

df = pd.read_csv("GameSales_dropped.csv")

# in the csv file it renamed Global_Sales; and this situation cause the error we fix this  with these lines
df.rename(columns={'Global_Sales;': 'Global_Sales'}, inplace=True)
df['Global_Sales'] = df['Global_Sales'].str.replace(';', '').astype(float)

# for eliminating nan values
df.dropna(inplace=True)

feature_names = ["Year","NA_Sales","EU_Sales","JP_Sales","Other_Sales","Global_Sales"]

X = df[feature_names]
y = df["Platform"]  # target attribute

# splitting 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# classification
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier()
}

# fittings
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} accuracy: {accuracy}")

# visualization of decision tree
decision_tree_model = models["Decision Tree"]
plt.figure(figsize=(16, 12))
plot_tree(decision_tree_model, feature_names=X.columns, class_names=np.unique(y).astype('str'), filled=True)
plt.savefig('decision_tree.pdf')  # Save the decision tree plot as a PDF
plt.close()  # Close the plot to prevent it from being displayed

# performance comparison
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.xlabel('Classification Algorithms')
plt.ylabel('Accuracy')
plt.title('Performance Comparison of Classification Algorithms')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.savefig('performance_comparison.pdf')  # Save the performance comparison plot as a PDF
plt.close()  # Close the plot to prevent it from being displayed
