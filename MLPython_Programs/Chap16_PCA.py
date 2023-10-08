### Chapter 16 Principal Component Analysis 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from mpl_toolkits import mplot3d

### PCA with the audiometric data

audiometric = pd.read_csv('audiometric.csv')
audiometric.shape
audiometric.head()

pd.options.display.max_columns = 10
round(audiometric.corr(), 2)

np.mean(audiometric, axis=0)
np.std(audiometric, axis=0)

scaler = StandardScaler()
scaler.fit(audiometric)
X = scaler.transform(audiometric)

np.mean(X, axis=0)
np.std(X, axis=0)

model = PCA()
model.fit(X)

model.explained_variance_

plt.plot(model.explained_variance_, 'o-')
plt.axhline(model.explained_variance_[3], color='k', linestyle='--', linewidth=1)
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('Scree Plot')

model.explained_variance_ratio_

plt.plot(model.explained_variance_ratio_, 'o-')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.title('PVE')

plt.plot(model.explained_variance_ratio_.cumsum(), 'o-')
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Proportion of Variance Explained')
plt.axhline(0.9, color='k', linestyle='--', linewidth=1)
plt.title('Cumulative PVE')

model.components_
columns = ['PC' + str(i) for i in range(1, 9)]
columns
pca_loadings = pd.DataFrame(model.components_.T, index=audiometric.columns, columns=columns)
round(pca_loadings, 2)

# Visualize pca loadings

fig, ax = plt.subplots(2, 2)
plt.subplots_adjust(hspace=1, wspace=0.5)   
for i in range(1, 5):
    ax = plt.subplot(2, 2, i)
    ax.plot(pca_loadings['PC' + str(i)], 'o-')
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xticks(range(8))
    ax.set_xticklabels(audiometric.columns, rotation=30)
    ax.set_title('PCA Loadings for PC' + str(i))

# PCA Scores

pca_scores = model.transform(X)
pca_scores = pd.DataFrame(pca_scores, columns=columns)
pca_scores.shape
pca_scores.head()

# visualize pca scores via biplot

sns.scatterplot(x='PC1', y='PC2', data=pca_scores)
plt.title('Biplot')

# Visualize pca scores via triplot
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_scores['PC1'], pca_scores['PC2'], pca_scores['PC3'], c='b')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

from sklearn.cluster import KMeans
model = KMeans(n_clusters=3, random_state=1, n_init=20)
model.fit(X)
model.labels_

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_scores['PC1'], pca_scores['PC2'], pca_scores['PC3'],
           c=model.labels_, cmap='rainbow')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# PCA without standardization

model = PCA()
model.fit(audiometric)
pca_loadings = pd.DataFrame(model.components_.T, index=audiometric.columns, columns=columns)
round(pca_loadings, 2)


### Principal Component Regression

growth = pd.read_csv('growth.csv')
growth.shape
growth.head(3)
growth.tail(3)

growth.index = growth['Quarter']
growth = growth.drop(columns=['Quarter'])

# Correlation between HK's growth rate and other countries
growth.corr().iloc[:, 0]

X_train = growth.iloc[:44, 1:]
X_train.shape

X_test = growth.iloc[44:, 1:]
X_test.shape

y_train = growth.iloc[:44, 0]

y_test = growth.iloc[44:, 0]

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

np.mean(X_train, axis=0)
np.std(X_train, axis=0)

np.mean(X_test, axis=0)
np.std(X_test, axis=0)

scores_mse = []
for k in range(1, 24):
    model = PCA(n_components=k)
    model.fit(X_train)
    X_train_pca = model.transform(X_train)
    loo = LeaveOneOut()
    mse = -cross_val_score(LinearRegression(), X_train_pca, y_train, 
                           cv=loo, scoring='neg_mean_squared_error')
    scores_mse.append(np.mean(mse))
min(scores_mse)
index = np.argmin(scores_mse)
index

plt.plot(range(1, 24), scores_mse)
plt.axvline(index + 1, color='k', linestyle='--', linewidth=1)
plt.xlabel('Number of Components')
plt.ylabel('Mean Squared Error')
plt.title('Leave-one-out Cross-validation Error')
plt.tight_layout()

model = PCA(n_components = index + 1)
model.fit(X_train)

X_train_pca = model.transform(X_train)
X_test_pca = model.transform(X_test)

reg = LinearRegression()
reg.fit(X_train_pca, y_train)

X_pca = np.vstack((X_train_pca, X_test_pca))
X_pca.shape

pred = reg.predict(X_pca)

y = growth.iloc[:, 0]

plt.figure(figsize=(10, 5))
ax = plt.gca()
plt.plot(y, label='Actual', color='k')
plt.plot(pred, label='Predicted', color='k', linestyle='--')
plt.xticks(range(1, 62))
ax.set_xticklabels(growth.index, rotation=90)
plt.axvline(44, color='k', linestyle='--', linewidth=1)
plt.xlabel('Quarter')
plt.ylabel('Growth Rate')
plt.title("Economic Growth of HongKong_CN")
plt.legend(loc='upper left')
plt.tight_layout()


