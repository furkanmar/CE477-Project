import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the dataset
file_path = 'GameSales_dropped.csv'
data = pd.read_csv(file_path)

# Encode categorical features
label_encoders = {}
for column in ['Platform', 'Genre', 'Publisher']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))
    label_encoders[column] = le

data = data.dropna()

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(data.drop(columns=['Name']))
feature_names = data.drop(columns=['Name']).columns

# Define the features to use for x and y axes in the plots
x_feature = 'Platform'
y_feature = 'Publisher'
z_feature ='Genre'

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
data['KMeans_Cluster'] = kmeans.fit_predict(X)
kmeans_silhouette = silhouette_score(X, data['KMeans_Cluster'])

# Report the size of each cluster
kmeans_sizes = data['KMeans_Cluster'].value_counts()
print("K-Means Cluster Sizes:")
print(kmeans_sizes)

# Report the cluster centroids
kmeans_centroids = scaler.inverse_transform(kmeans.cluster_centers_)
print("K-Means Cluster Centroids:")
print(kmeans_centroids)

# Visualize the K-Means clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data[x_feature], y=data[y_feature], hue=data['KMeans_Cluster'], palette='viridis')
plt.title('K-Means Clustering')
plt.xlabel(x_feature)
plt.ylabel(y_feature)
plt.savefig('kmeans_clusters.pdf')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data[x_feature], y=data[z_feature], hue=data['KMeans_Cluster'], palette='viridis')
plt.title('K-Means Clustering')
plt.xlabel(x_feature)
plt.ylabel(z_feature)
plt.savefig('kmeans_clusters1.pdf')
plt.show()

# Apply hierarchical clustering with single linkage
hc_single = AgglomerativeClustering(n_clusters=3, linkage='single')
data['HC_Single_Cluster'] = hc_single.fit_predict(X)
hc_single_silhouette = silhouette_score(X, data['HC_Single_Cluster'])

# Draw the dendrogram for single linkage
plt.figure(figsize=(10, 6))
Z_single = linkage(X, method='single')
dendrogram(Z_single)
plt.title('Hierarchical Clustering Dendrogram (Single Linkage)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.savefig('hc_single_dendrogram.pdf')
plt.show()

# Apply hierarchical clustering with complete linkage
hc_complete = AgglomerativeClustering(n_clusters=3, linkage='complete')
data['HC_Complete_Cluster'] = hc_complete.fit_predict(X)
hc_complete_silhouette = silhouette_score(X, data['HC_Complete_Cluster'])

# Draw the dendrogram for complete linkage
plt.figure(figsize=(10, 6))
Z_complete = linkage(X, method='complete')
dendrogram(Z_complete)
plt.title('Hierarchical Clustering Dendrogram (Complete Linkage)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.savefig('hc_complete_dendrogram.pdf')
plt.show()


dbscan = DBSCAN(eps=0.5, min_samples=5)
data['DBSCAN_Cluster'] = dbscan.fit_predict(X)
# DBSCAN may produce noise points labeled as -1; filter these out for silhouette score calculation
dbscan_silhouette = silhouette_score(X[data['DBSCAN_Cluster'] != -1], data['DBSCAN_Cluster'][data['DBSCAN_Cluster'] != -1])


dbscan_sizes = data['DBSCAN_Cluster'].value_counts()
print("DBSCAN Cluster Sizes:")
print(dbscan_sizes)

# Visualize the DBSCAN clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data[x_feature], y=data[y_feature], hue=data['DBSCAN_Cluster'], palette='viridis')
plt.title('DBSCAN Clustering')
plt.xlabel(x_feature)
plt.ylabel(y_feature)
plt.savefig('dbscan_clusters.pdf')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data[x_feature], y=data[z_feature], hue=data['DBSCAN_Cluster'], palette='viridis')
plt.title('DBSCAN Clustering')
plt.xlabel(x_feature)
plt.ylabel(z_feature)
plt.savefig('dbscan_clusters1.pdf')
plt.show()

# Compare silhouette scores
performance = {
    'K-Means': kmeans_silhouette,
    'Hierarchical (Single Linkage)': hc_single_silhouette,
    'Hierarchical (Complete Linkage)': hc_complete_silhouette,
    'DBSCAN': dbscan_silhouette
}

print("Silhouette Scores Comparison:")
print(performance)

# Visualize silhouette scores comparison
plt.figure(figsize=(10, 6))
sns.barplot(x=list(performance.keys()), y=list(performance.values()))
plt.title('Silhouette Scores Comparison of Clustering Methods')
plt.ylabel('Silhouette Score')
plt.xlabel('Clustering Method')
plt.ylim(0, 1)
plt.savefig('silhouette_scores_comparison.pdf')
plt.show()