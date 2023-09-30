# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 20:10:54 2023

@author: ahmed el-aloul
student id: 301170922
ASSIGNMENT #2 - K-Means & DBSCAN Clustering
COMP 257 SECTION 401

"""
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, classification_report
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
    
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '1'

# 1. Load Olivetti faces dataset
faces = datasets.fetch_olivetti_faces()
X = faces.data
y = faces.target

# 2. Splitting the dataset
# Rationale: Splitting data into 60% training, 20% validation, and 20% test is a common practice.
# It provides a good balance between maximizing training data and having a solid amount to validate and test the model.
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=0)

# 3. k-fold cross-validation
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=StratifiedKFold(n_splits=5))
grid.fit(X_train, y_train)
y_pred_val = grid.predict(X_val)
print("Before K-Means")
print("Best parameters found: ", grid.best_params_)
print(classification_report(y_val, y_pred_val))

# 4. K-Means for dimensionality reduction
# Rationale for similarity measure: K-Means uses Euclidean distance to measure the similarity. 
# It's the default and most commonly used measure for K-Means because of its geometric properties. 
# For high dimensional data like images, it can be sensitive to irrelevant features.
silhouette_scores = []
k_range = range(2, 15)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_train)
    cluster_labels = kmeans.predict(X_train)
    score = silhouette_score(X_train, cluster_labels)
    silhouette_scores.append(score)

best_k = k_range[np.argmax(silhouette_scores)]
kmeans = KMeans(n_clusters=best_k, random_state=0).fit(X_train)
X_train_reduced = kmeans.transform(X_train)
X_val_reduced = kmeans.transform(X_val)

# 5. Train a classifier on reduced set
clf_kmeans = SVC(kernel='rbf', C=grid.best_params_['C'], gamma=grid.best_params_['gamma']).fit(X_train_reduced, y_train)
y_pred_val_kmeans = clf_kmeans.predict(X_val_reduced)
print("\nAfter K-Means")
print(classification_report(y_val, y_pred_val_kmeans))

# 6. DBSCAN for clustering
# Rationale for similarity measure: DBSCAN measures similarity based on density rather than distance, 
# which can be beneficial for image data as facial structures can have localized features with high density. 
# Euclidean distance, used in conjunction with density, helps in clustering these dense areas together.

# Preprocess for DBSCAN (using a smaller dataset due to computational limitations)
small_sample_size = 100
small_sample_indices = np.random.choice(len(X_train), small_sample_size, replace=False)
X_small_sample = X_train[small_sample_indices]

scaler = StandardScaler().fit(X_small_sample)
X_scaled = scaler.transform(X_small_sample)

neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X_scaled)
distances, indices = nbrs.kneighbors(X_scaled)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
eps_value = distances[np.argmax(np.diff(distances)) + 1]

dbscan = DBSCAN(eps=eps_value, min_samples=5).fit(X_scaled)
labels = dbscan.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print("Estimated number of clusters using DBSCAN:", n_clusters_)

# Visualization
pca_2d = PCA(n_components=2).fit_transform(X_scaled)
plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=labels, cmap='rainbow')
plt.title('DBSCAN Clusters using PCA for 2D visualization')
plt.colorbar()
plt.show()
