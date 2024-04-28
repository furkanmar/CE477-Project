import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from tabulate import tabulate
import pdfkit

df = pd.read_csv('GameSales.csv')
target_attribute = "Genre"

#in the csv file it renamed Global_Sales; and this situation cause the error we fix this  with these lines
df.rename(columns={'Global_Sales;': 'Global_Sales'}, inplace=True)
df['Global_Sales'] = df['Global_Sales'].str.replace(';', '').astype(float)

#for elimiate nan values
df.dropna(inplace=True)

#print(df["Genre"].unique())

#print(df.groupby('Genre').size())


feature_names = ["Name","Platform","Publisher"]
X = df[feature_names]
y = df["Genre"]

#One-Hot encoding
encoder = OneHotEncoder(drop='first', sparse=False)
X_encoded = encoder.fit_transform(X)
encoded_feature_names = encoder.get_feature_names_out(feature_names)
X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_feature_names)

X_train, X_test, y_train, y_test = train_test_split(X_encoded_df, y, random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Accuracy of Logistic regression (training set): {:.2f}'.format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression (test set): {:.2f}'.format(logreg.score(X_test, y_test)))

clf = DecisionTreeClassifier().fit(X_train, y_train)
print("Accuracy of Decision Tree(training set): {:.2f}".format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree (test set): {:.2f}'.format(clf.score(X_test, y_test)))

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print("Accuracy of KNN (training set): {:.2f}".format(knn.score(X_train, y_train)))
print("Accuracy of KNN (test set): {:.2f}".format(knn.score(X_test,y_test)))

#making table from the data
mydata = [
    ["Logistic Regression",logreg.score(X_train, y_train),logreg.score(X_test, y_test)],
    ["Decision Tree",clf.score(X_train, y_train),clf.score(X_test, y_test)],
    ["KNN",knn.score(X_train, y_train),knn.score(X_test, y_test)]
]
head =["Classifiers", "Test Score", "Train Score"]
title = "Classifier Comparison"
html_table = f"<h2>{title}</h2>" + tabulate(mydata, headers=head, tablefmt="html")
with open("table.html", "w") as f:
    f.write(html_table)

plt.figure(figsize=(15, 10))
plot_tree(clf, feature_names=encoded_feature_names, class_names=clf.classes_, filled=True)
plt.savefig("decision_tree_plot.pdf")

# Convert HTML table to PDF
pdfkit.from_file('table.html', 'classifier_comparison.pdf')

# Close the plots
plt.close()