### Chapter 17 Cluster Analysis 


# 17.5 Example of Correlation-based Distance Measure

import matplotlib.pyplot as plt
import seaborn as sns

iris = sns.load_dataset('iris')
iris50 = iris.iloc[50, :-1]
iris99 = iris.iloc[99, :-1]
iris99_2 = 2 * iris99

fig = plt.figure(figsize=(10, 6))
plt.plot(iris50, 'ko-', label='Observation 50')
plt.plot(iris99, 'bo-', label='Observation 99')
plt.plot(iris99_2,'ko--', label='Hypothetical Observation')
plt.title('Observation Profile')
plt.legend()
plt.tight_layout()

# 17.6 K-Means Clustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

### Simulated data

np.random.seed(1)
cluster1 = np.random.normal(0, 1, 100).reshape(-1, 2)
cluster1 = pd.DataFrame(cluster1, columns=['x1', 'x2'])
cluster1['y'] = 0

cluster1.head()

np.random.seed(10)
cluster2 = np.random.normal(3, 1, 100).reshape(-1, 2)
cluster2 = pd.DataFrame(cluster2, columns=['x1', 'x2'])
cluster2['y'] = 1

cluster2.head()

data = pd.concat([cluster1, cluster2])
data.shape

sns.scatterplot(x='x1', y='x2', data=cluster1, color='k', label='y=0')
sns.scatterplot(x='x1', y='x2', data=cluster2, color='b', label='y=1')
plt.legend()
plt.title('True Clusters (K=2)')

# K-means clustering with K=2

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

model = KMeans(n_clusters=2, random_state=123, n_init=20)
model.fit(X)
model.labels_
model.cluster_centers_
model.inertia_

pd.crosstab(y, model.labels_, rownames=['Actual'], colnames=['Predicted'])

sns.scatterplot(x='x1', y='x2', data=data[model.labels_ == 0], color='k', label='y=0')
sns.scatterplot(x='x1', y='x2', data=data[model.labels_ == 1], color='b', label='y=1')
plt.legend()
plt.title('Estimated Clusters (K=2)')

# K-means clustering with K=3

model = KMeans(n_clusters=3, random_state=2, n_init=20)
model.fit(X)
model.labels_

sns.scatterplot(x='x1', y='x2', data=data[model.labels_ == 0], color='k', label='y=0')
sns.scatterplot(x='x1', y='x2', data=data[model.labels_ == 1], color='b', label='y=1')
sns.scatterplot(x='x1', y='x2', data=data[model.labels_ == 2], color='cornflowerblue', label='y=2')
plt.legend()
plt.title('Estimated Clusters (K=3)')

# K-means clustering with K=4

model = KMeans(n_clusters=4, random_state=3, n_init=20)
model.fit(X)
model.labels_

sns.scatterplot(x='x1', y='x2', data=data[model.labels_ == 0], color='b', label='y=0')
sns.scatterplot(x='x1', y='x2', data=data[model.labels_ == 1], color='cornflowerblue', label='y=1')
sns.scatterplot(x='x1', y='x2', data=data[model.labels_ == 2], color='darkblue', label='y=2')
sns.scatterplot(x='x1', y='x2', data=data[model.labels_ == 3], color='k', label='y=3')
plt.legend()
plt.title('Estimated Clusters (K=4)')

# Effect of n_init

model = KMeans(n_clusters=4, random_state=4, n_init=1)
model.fit(X)
model.inertia_

model = KMeans(n_clusters=4, random_state=4, n_init=30)
model.fit(X)
model.inertia_

model = KMeans(n_clusters=4, random_state=4, n_init=1000)
model.fit(X)
model.inertia_

# Choose optimal K by elbow method

sse = []
for k in range(1,16):
    model = KMeans(n_clusters=k, random_state=1, n_init=20)
    model.fit(X)
    sse.append(model.inertia_)
print(sse)

plt.plot(range(1, 16), sse, 'o-')
plt.axhline(sse[1], color='k', linestyle='--', linewidth=1)
plt.xlabel('K')
plt.ylabel('SSE')
plt.title('K-means Clustering')

# Choose optimal K by AIC

aic = sse + 2 * 2 * np.arange(1, 16)
aic
min(aic)
np.argmin(aic)

plt.plot(range(1, 16), aic, 'o-')
plt.axvline(np.argmin(aic) + 1, color='k', linestyle='--', linewidth=1)
plt.xlabel('K')
plt.ylabel('AIC')
plt.title('K-means Clustering')

# Choose optimal K by BIC

bic = sse + 2 * np.log(100) * np.arange(1, 16)
bic
min(bic)
np.argmin(bic)

plt.plot(range(1, 16), bic, 'o-')
plt.axvline(np.argmin(bic) + 1, color='k', linestyle='--', linewidth=1)
plt.xlabel('K')
plt.ylabel('BIC')
plt.title('K-means Clustering')


### K-means clustering with the iris data

iris = sns.load_dataset('iris')
iris.head()

Dict = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
iris.species = iris.species.map(Dict)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(iris['sepal_length'], iris['sepal_width'],
           iris['petal_length'], c=iris['species'], cmap='rainbow')
ax.set_xlabel('sepal_length')
ax.set_ylabel('sepal_width')
ax.set_zlabel('petal_length')

# K-means clustering with K=3

X3= iris.iloc[:, :3]
model = KMeans(n_clusters=3, random_state=1, n_init=20)

model.fit(X3)
model.labels_

labels = pd.DataFrame(model.labels_, columns=['label'])
d = {0: 2, 1: 0, 2: 1}
pred = labels.label.map(d)

table = pd.crosstab(iris.species, pred, rownames=['Actual'], colnames=['Predicted'])
table

accuracy = np.trace(table) / len(iris)
accuracy

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(iris['sepal_length'], iris['sepal_width'],
           iris['petal_length'], c=pred, cmap='rainbow')
ax.set_xlabel('sepal_length')
ax.set_ylabel('sepal_width')
ax.set_zlabel('petal_length')

### Hierarchical Clustering with the iris data

X = iris.iloc[:, :-1]
X.shape

# Complete linkage

linkage_matrix = linkage(X, 'complete')
linkage_matrix.shape
dendrogram(linkage_matrix)
plt.title('Complete Linkage')

model = AgglomerativeClustering(n_clusters=3, linkage='complete')
model.fit(X)
model.labels_

labels = pd.DataFrame(model.labels_, columns=['label'])
d = {0: 2, 1: 0, 2: 1}
pred = labels['label'].map(d)

table = pd.crosstab(iris.species, pred, rownames=['Actual'], colnames=['Predicted'])
table

accuracy = np.trace(table) / len(iris)
accuracy

# Average linkage

linkage_matrix = linkage(X, 'average')
dendrogram(linkage_matrix)
plt.title('Average Linkage')

model = AgglomerativeClustering(n_clusters=3, linkage='average')

model.fit(X)
model.labels_

labels = pd.DataFrame(model.labels_, columns=['label'])
d = {0: 1, 1: 0, 2: 2}
pred = labels['label'].map(d)

table = pd.crosstab(iris.species, pred, rownames=['Actual'], colnames=['Predicted'])
table

accuracy = np.trace(table) / len(iris)
accuracy

# Single linkage

linkage_matrix = linkage(X, 'single')
dendrogram(linkage_matrix)
plt.title('Single Linkage')

model = AgglomerativeClustering(n_clusters=3, linkage='single')

model.fit(X)
model.labels_

labels = pd.DataFrame(model.labels_, columns=['label'])
d = {0: 1, 1: 0, 2: 2}
pred = labels['label'].map(d)

table = pd.crosstab(iris.species, pred, rownames=['Actual'], colnames=['Predicted'])
table

accuracy = np.trace(table) / len(iris)
accuracy

# Centroid linkage

linkage_matrix = linkage(X, 'centroid')
dendrogram(linkage_matrix)
plt.title('Centroid Linkage')

labels = fcluster(linkage_matrix, t=3, criterion='maxclust')
labels

labels = pd.DataFrame(labels, columns=['label'])
d = {1: 0, 2: 2, 3: 1}
pred = labels['label'].map(d)

table = pd.crosstab(iris.species, pred, rownames=['Actual'], colnames=['Predicted'])
table

accuracy = np.trace(table) / len(iris)
accuracy

## Average linkage for standardized data

scaler = StandardScaler()
scaler.fit(X)
X_s = scaler.transform(X)

model = AgglomerativeClustering(n_clusters=3, linkage='average')
model.fit(X_s)
model.labels_

table = pd.crosstab(iris.species, model.labels_, rownames=['Actual'], colnames=['Predicted'])
table

accuracy = np.trace(table) / len(iris)
accuracy

## Correlation-based distance measure

dist_matrix = 1 - np.corrcoef(X) 
dist_matrix.shape
dist_matrix[:3, :3]

sns.clustermap(dist_matrix, cmap='Blues')

dist = squareform(dist_matrix, checks=False)

dist.shape

linkage_matrix = linkage(dist, 'centroid')
dendrogram(linkage_matrix)
plt.title('Correlated-based Centroid Linkage')

labels = fcluster(linkage_matrix, t=3, criterion='maxclust')
labels

labels = pd.DataFrame(labels, columns=['label'])
d = {1: 0, 2: 2, 3: 1}
pred = labels['label'].map(d)

table = pd.crosstab(iris.species, pred, rownames=['Actual'], colnames=['Predicted'])
table

accuracy = np.trace(table) / len(iris)
accuracy
