import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# import plotly.express as px
import pandas as pd
import json


# Load feature vectors
feature_vectors = torch.load(
    'feature_vectors.pt', map_location=torch.device('cpu'))
X = feature_vectors.numpy()

print(feature_vectors.shape)

# K-means clustering
n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
cluster_labels = kmeans.fit_predict(X)

# tsne = TSNE(n_components=2, random_state=0)  # Reduce to 2 dimensions for visualization
# X_reduced = tsne.fit_transform(X)

# # visualization
# df = pd.DataFrame(X_reduced, columns=['t-SNE-1', 't-SNE-2'])
# df['Cluster'] = cluster_labels.astype(str)
# df['Index'] = df.index

# # Plot using Plotly Express
# fig = px.scatter(
#     df, x='t-SNE-1', y='t-SNE-2',
#     color='Cluster', hover_data=['Index'],
#     title='K-means Clustering with t-SNE Reduced Data')
# fig.show()

# use PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
X_reduced = pca.fit_transform(X)

# visualization
df = pd.DataFrame(X_reduced, columns=['PCA-1', 'PCA-2'])
df['Cluster'] = cluster_labels.astype(str)
df['Index'] = df.index

# # Plot using Plotly Express
# fig = px.scatter(
#     df, x='PCA-1', y='PCA-2',
#     color='Cluster', hover_data=['Index'],
#     title='K-means Clustering with PCA Reduced Data')

# # save the plot
# fig.write_html('cluster_visualisation.html')


# save clustering results
# Create a dictionary to hold indices for each cluster
clusters_dict = {i: [] for i in range(n_clusters)}

# Populate the dictionary with indices
for index, label in enumerate(cluster_labels):
    clusters_dict[label].append(index)

# Now, 'clusters_dict' contains the indices of data points for each cluster
# Save the dictionary to a file
with open('clusters.json', 'w') as f:
    json.dump(clusters_dict, f)