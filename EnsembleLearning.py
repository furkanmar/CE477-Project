import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load the dataset
file_path = 'GameSales_dropped.csv'
data = pd.read_csv(file_path)

# Encode categorical features
label_encoders = {}
for column in ['Platform', 'Publisher']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))
    label_encoders[column] = le

# Drop rows with missing values (if any)
data = data.dropna()

# Define features and target
X = data.drop(columns=['Genre', 'Name'])
y = data['Genre']

# Encode target variable 'Genre'
le_genre = LabelEncoder()
y = le_genre.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Bagging classifier
bagging_clf = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
bagging_clf.fit(X_train, y_train)
y_pred_bagging = bagging_clf.predict(X_test)
bagging_accuracy = accuracy_score(y_test, y_pred_bagging)

# Initialize and train AdaBoost classifier
adaboost_clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, algorithm='SAMME', random_state=42)
adaboost_clf.fit(X_train, y_train)
y_pred_adaboost = adaboost_clf.predict(X_test)
adaboost_accuracy = accuracy_score(y_test, y_pred_adaboost)

# Initialize and train Random Forest classifier
rf_clf = RandomForestClassifier(n_estimators=50, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

# Detailed performance metrics for Bagging
bagging_precision = precision_score(y_test, y_pred_bagging, average='weighted', zero_division=0)
bagging_recall = recall_score(y_test, y_pred_bagging, average='weighted')
bagging_f1 = f1_score(y_test, y_pred_bagging, average='weighted')
bagging_classification_report = classification_report(y_test, y_pred_bagging, target_names=le_genre.classes_, zero_division=0)

# Cross-validation scores for Bagging
bagging_cross_val_scores = cross_val_score(bagging_clf, X, y, cv=5, scoring='accuracy')

# Compare the performance
performance = {
    'Bagging': bagging_accuracy,
    'AdaBoost': adaboost_accuracy,
    'Random Forest': rf_accuracy
}

# Plotting performance comparison
plt.figure(figsize=(10, 6))
sns.barplot(x=list(performance.keys()), y=list(performance.values()))
plt.title('Performance Comparison of Ensemble Methods')
plt.ylabel('Accuracy')
plt.xlabel('Algorithm')
plt.ylim(0, 1)
plt.savefig('performance_comparison')
plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {title}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('Confusion Matrix ')
    plt.show()

# Plot confusion matrices
plot_confusion_matrix(y_test, y_pred_bagging, 'Bagging', 'confusion_matrix_bagging.pdf')
plot_confusion_matrix(y_test, y_pred_adaboost, 'AdaBoost', 'confusion_matrix_adaboost.pdf')
plot_confusion_matrix(y_test, y_pred_rf, 'Random Forest', 'confusion_matrix_rf.pdf')

# Plot cross-validation results
plt.figure(figsize=(10, 6))
sns.boxplot(data=bagging_cross_val_scores)
plt.title('Cross-Validation Accuracy Scores for Bagging')
plt.ylabel('Accuracy')
plt.xlabel('Cross-Validation Fold')
plt.savefig('cross_val_scores_bagging.pdf')
plt.show()

# Print detailed analysis for Bagging
print("Bagging Detailed Analysis:")
print(f"Accuracy: {bagging_accuracy:.4f}")
print(f"Precision: {bagging_precision:.4f}")
print(f"Recall: {bagging_recall:.4f}")
print(f"F1 Score: {bagging_f1:.4f}")
print("\nClassification Report:")
print(bagging_classification_report)
print("Cross-Validation Scores:", bagging_cross_val_scores)
print("Mean Cross-Validation Score:", bagging_cross_val_scores.mean())

print("\nOverall Performance Comparison:")
print(performance)