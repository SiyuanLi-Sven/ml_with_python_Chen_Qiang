### Chapter 10 KNN 

## KNN Regression for Motorcycle Data 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor

mcycle = pd.read_csv('mcycle.csv')
mcycle.shape

mcycle.head()
mcycle.describe()

sns.scatterplot(x='times', y='accel', data=mcycle)
plt.title('Simulated Motorcycle Accident')

X_raw = np.array(mcycle.times)

X = np.array(mcycle.times).reshape(-1, 1)

y = mcycle.accel 

fig, ax = plt.subplots(2, 2, figsize=(9,6), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.5, wspace=0.5)
for i,k in zip([1, 2, 3, 4], [1, 10, 25, 50]):
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X, y)
    pred = model.predict(np.arange(60).reshape(-1, 1))
    plt.subplot(2, 2, i)
    sns.scatterplot(x='times', y='accel', s=20, data=mcycle, facecolor='none', edgecolor='k')
    plt.plot(np.arange(60), pred, 'b')
    plt.text(0, 55, f'K = {k}')
plt.tight_layout()
    
list(zip([1, 2, 3, 4], [1, 10, 25, 50]))

## Decision boundary for KNN on iris data

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions

X, y = load_iris(return_X_y=True)
X2 = X[:, 2:4]

# The plot below is time-consuming

fig, ax = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.1, wspace=0.1)
for i, k in zip([1, 2, 3, 4], [1, 10, 25, 50]):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X2, y)
    plt.subplot(2, 2, i)
    plot_decision_regions(X2, y, model)
    plt.xlabel('petal_length')
    plt.ylabel('petal_width')
    plt.text(0.3, 3, f'K = {k}')
plt.tight_layout()


## KNN Classifier for Winsconsin Breast Cancer Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

cancer = load_breast_cancer()

df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['diagnosis'] = cancer.target

d = {0: 'malignant', 1: 'benign'}
df['diagnosis'] = df['diagnosis'].map(d)

df.shape

pd.options.display.max_columns = 40 
df.head(2)

df.iloc[:,:3].describe()

df.diagnosis.value_counts()
df.diagnosis.value_counts(normalize=True)

sns.boxplot(x='diagnosis', y='mean radius', data=df)

# Split samples 

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=100, random_state=1)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

np.mean(X_train_s, axis=0)
np.std(X_train_s, axis=0)
np.mean(X_test_s, axis=0)
np.std(X_test_s, axis=0)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)
pred = model.predict(X_test_s)
pred

pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])


# Choose optimal K via test set

scores = []
ks = range(1, 51)
for k in ks:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_s, y_train)
    score = model.score(X_test_s, y_test)
    scores.append(score)
max(scores)
index_max = np.argmax(scores)
print(f'Optimal K: {ks[index_max]}')

# Graph accuracy versus K 

plt.plot(ks, scores, 'o-')
plt.xlabel('K')
plt.axvline(ks[index_max], linewidth=1, linestyle='--', color='k')
plt.ylabel('Accuracy')
plt.title('KNN')
plt.tight_layout()

# Graph error rate versus K

errors = 1 - np.array(scores)
plt.plot(ks, errors, 'o-')
plt.xlabel('K')
plt.axvline(ks[index_max], linewidth=1, linestyle='--', color='k')
plt.ylabel('Error Rate')
plt.title('KNN')
plt.tight_layout()

# Graph error rate versus 1/K

errors = 1 - np.array(scores)
ks_inverse = 1 / np.array(ks)
plt.plot(ks_inverse, errors, 'o-')
plt.xlabel('1/K')
plt.ylabel('Error Rate')
plt.title('KNN')
plt.tight_layout()

# Choose optimal K via CV

param_grid = {'n_neighbors': range(1, 51)}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(KNeighborsClassifier(), param_grid, cv=kfold)
model.fit(X_train_s, y_train)

model.best_params_
model.score(X_test_s, y_test)


    
    