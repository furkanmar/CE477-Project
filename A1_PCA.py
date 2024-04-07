import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('GameSales.csv')
df_cleaned = df.dropna()
nonNumericCol = ['Name', 'Platform', 'Genre', 'Publisher']  #take the non-numeric attribures
nonNumericData = df_cleaned[nonNumericCol]

nonNumericOHC = pd.get_dummies(nonNumericData) #ONE HOT ENCODING
df_numeric = df_cleaned.drop(nonNumericCol, axis=1) #DROP FROM ORIGINAL COLUMNS
df_encoded = pd.concat([df_numeric, nonNumericOHC], axis=1) #connect ohc and numeric data
standardize = StandardScaler()
scaledData = standardize.fit_transform(df_encoded)

#PCA
pca = PCA(n_components=3)
pcaResult = pca.fit_transform(scaledData)
pca_df = pd.DataFrame(data=pcaResult, columns=['PC1', 'PC2', 'PC3' ]) #DATA FRAME FROM PCA results.

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], alpha=0.5)
ax.set_xlabel('Principal Component Number 1')
ax.set_ylabel('Principal Component Number 2')
ax.set_zlabel('Principal Component Number 3')
ax.set_title('PCA of 3D\n')
plt.show()

