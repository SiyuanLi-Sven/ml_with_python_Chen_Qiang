### Chapter 15  Artificial Neural Network 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance
from sklearn.inspection import plot_partial_dependence
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from mlxtend.plotting import plot_decision_regions

# Feedforward Network with Boston Housing Data

Boston = load_boston()
X = pd.DataFrame(Boston.data, columns=Boston.feature_names)
y = Boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

np.min(X_train_s, axis=0)
np.max(X_train_s, axis=0)

np.min(X_test_s, axis=0)
np.max(X_test_s, axis=0)

# Single hidden layer 

model = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(5,), random_state=123, max_iter=10000)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)

model.intercepts_
model.coefs_
model.n_iter_

table = pd.DataFrame(model.coefs_[0], index=Boston.feature_names, columns=[1,2,3,4,5])

sns.heatmap(table, cmap='Blues', annot=True)
plt.xlabel('Neuron')
plt.title('Neural Network Weights')
plt.tight_layout()

sns.heatmap(abs(table), cmap='Blues', annot=True)
plt.xlabel('Neuron')
plt.title('Neural Network Weights (Absolute Values)')
plt.tight_layout()

# Permutation Importance Plot

result = permutation_importance(model, X_test_s, y_test, n_repeats=20, random_state=42)

dir(result)

index = result.importances_mean.argsort()
plt.boxplot(result.importances[index].T, vert=False, labels=X_test.columns[index])
plt.title('Permutation Importances')

mean_importance = pd.DataFrame(result.importances_mean[index], index=Boston.feature_names[index], columns=['mean_importance'])
mean_importance

# Partial Dependence Plot

X_train_s = pd.DataFrame(X_train_s, columns=Boston.feature_names)
plot_partial_dependence(model, X_train_s, ['LSTAT', 'RM'])

## Test Set R2 vs Number of Neurons

scores = []
for n_neurons in range(1, 41):
    model = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(n_neurons,), random_state=123, max_iter=10000)
    model.fit(X_train_s, y_train)
    score = model.score(X_test_s, y_test)
    scores.append(score)

index = np.argmax(scores)
range(1, 41)[index]

plt.plot(range(1, 41), scores, 'o-')
plt.axvline(range(1, 41)[index], linestyle='--', color='k', linewidth=1)
plt.xlabel('Number of Nodes')
plt.ylabel('R2')
plt.title('Test Set R2 vs Number of Nodes')

""" Choose number of nodes by CV
param_grid = {'hidden_layer_sizes':[(5,),(10,),(20,)]}
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(MLPRegressor(solver='lbfgs', random_state=123, max_iter=100000), param_grid, cv=kfold)
model.fit(X_train_s,y_train)
model.best_params_
model.score(X_test_s,y_test)
"""

# Best single layer model

model = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(19,), random_state=123, max_iter=10000)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)

plot_partial_dependence(model, X_train_s, ['LSTAT', 'RM'])

# Two hidden layers

model = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(5, 5), random_state=123, max_iter=10000)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)

best_score = 0
best_sizes = (1, 1)
for i in range(1, 11):
    for j in range(1, 11):
        model = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(i, j), random_state=123, max_iter=10000)
        model.fit(X_train_s, y_train)
        score = model.score(X_test_s, y_test)
        if best_score < score:
            best_score = score
            best_sizes = (i, j)
best_score
best_sizes
     
## ANN for Binary Classification with the Spam data

Spam = pd.read_csv('spam.csv')
Spam.shape
Spam.head()
Spam.spam.value_counts(normalize=True)

X = Spam.iloc[:, :-1]
y = Spam.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

model = MLPClassifier(hidden_layer_sizes=(100,), random_state=123, max_iter=10000)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)
model.n_iter_

model = MLPClassifier(hidden_layer_sizes=(500, 500), random_state=123, max_iter=10000)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)
model.n_iter_

# Regularization via Early stopping

model = MLPClassifier(hidden_layer_sizes=(500, 500), random_state=123, early_stopping=True, validation_fraction=0.25, max_iter=10000)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)
model.n_iter_

# Regularization via Weight decay

model = MLPClassifier(hidden_layer_sizes=(500,500), random_state=123, alpha=0.1, max_iter=10000)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)
model.n_iter_

prob = model.predict_proba(X_test_s)
prob[:3]

pred = model.predict(X_test_s)
pred[:3]

pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])

## ANN for binary classification with the simulated make_moons data

# Deterministic Data
X, y = make_moons(n_samples=100)
data = pd.DataFrame(X, columns=['x1', 'x2'])
sns.scatterplot(x='x1', y='x2', data=data, hue=y, palette=['blue', 'black'])

# Data with Noise
X, y = make_moons(n_samples=100, noise=0.3, random_state=0)
data = pd.DataFrame(X, columns=['x1', 'x2'])
sns.scatterplot(x='x1', y='x2', data=data, hue=y, palette=['blue', 'black'])

# Test Data
X_test, y_test = make_moons(n_samples=1100, noise=0.4, random_state=0)
X_test = X_test[100:, :]
y_test = y_test[100:]

model = MLPClassifier(hidden_layer_sizes=(100, 100), random_state=123, max_iter=10000)
model.fit(X, y)
model.score(X_test, y_test)

# Weight decay and decision boundary
  
plt.figure(figsize=(9, 6))    
for i, alpha in enumerate([0.01, 0.1, 1, 10]):
    model = MLPClassifier(hidden_layer_sizes=(100, 100), alpha=alpha, random_state=123, max_iter=10000)
    model.fit(X, y)
    plt.subplot(2, 2, i + 1)
    accuracy = model.score(X_test, y_test)
    plot_decision_regions(X, y, model, legend=0)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'alpha = {alpha}')
    plt.text(-2, 2, f'Accuracy = {accuracy}')
plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.tight_layout()

