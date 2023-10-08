### Chapter 14 Support Vector Machine 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.svm import LinearSVC
from sklearn.datasets import load_boston
from sklearn.datasets import load_digits
from sklearn.datasets import make_blobs
from mlxtend.plotting import plot_decision_regions

# Simulated Example

X, y = make_blobs(n_samples=40, centers=2, n_features=2, random_state=6)
y = 2 * y - 1
data = pd.DataFrame(X,columns=['x1', 'x2'])
sns.scatterplot(x='x1', y='x2', data=data, hue=y, palette=['blue','black'])

model = LinearSVC(C=1000, loss='hinge', random_state=123)
model.fit(X, y)

model.get_params()

dist = model.decision_function(X)
dist

index = np.where(y * dist <= (1 + 1e-10))
index

X[index]

def support_vectors(model, X, y):
    dist = model.decision_function(X)
    index = np.where(y * dist <= (1 + 1e-10))
    return X[index]

support_vectors(model, X, y)

def svm_plot(model,X,y):
    data = pd.DataFrame(X,columns=['x1', 'x2'])
    data['y'] = y
    sns.scatterplot(x='x1', y='x2', data=data, s=30, hue=y, palette=['blue','black'])
    s_vectors = support_vectors(model, X, y)
    plt.scatter(s_vectors[:, 0], s_vectors[:, 1], s=100,
                linewidth=1, facecolors='none', edgecolors='k')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])
    C = model.get_params()['C']
    plt.title(f'SVM (C = {C})')

svm_plot(model, X, y)

# LinearSVC implemented by liblinear     
# has more flexibility in the choice of penalties and loss functions 
# and should scale better to large numbers of samples.

model = LinearSVC(C=0.1, loss="hinge", random_state=123, max_iter=1e4)
model.fit(X, y)
support_vectors(model, X, y)
svm_plot(model, X, y)

# Choose optimal hyperparameter by CV

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
model = GridSearchCV(LinearSVC(loss="hinge", random_state=123, max_iter=1e4), param_grid, cv=kfold)
model.fit(X, y)
model.best_params_
model = model.best_estimator_
len(support_vectors(model, X, y))
model.intercept_
model.coef_
svm_plot(model, X, y)

""" Exercise

param_grid = {'C': np.linspace(0.001,0.1,100)}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
model = GridSearchCV(LinearSVC(loss="hinge", random_state=123, max_iter=1e4),param_grid,cv=kfold)
model.fit(X, y)
model.best_params_
model = model.best_estimator_
len(support_vectors(model, X, y))
svm_plot(model,X,y)

"""

# Evalutae prediction performance

X_test, y_test = make_blobs(n_samples=1040, centers=2, n_features=2, random_state=6)
y_test = 2 * y_test - 1
X_test = X_test[40:, :]
y_test = y_test[40:]

data_test = pd.DataFrame(X_test, columns=['x1','x2'])
sns.scatterplot(x='x1', y='x2', data=data_test, hue=y_test, palette=['blue','black'])

model.score(X_test, y_test)

pred = model.predict(X_test)
pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])


# Alternative estimation via SVC implemented by libsvm

model = SVC(kernel='linear', C=0.01, random_state=123)
model.fit(X, y)

model.n_support_
model.support_
model.support_vectors_
model.intercept_
model.coef_

model.score(X_test, y_test)
svm_plot(model, X, y)


## Nonlinear Decision Boundary

np.random.seed(1)
X = np.random.randn(200, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
y = np.where(y, 1, -1)

data = pd.DataFrame(X, columns=['x1', 'x2'])
sns.scatterplot(x='x1', y='x2', data=data, hue=y, palette=['blue','black'])

model = SVC(kernel='rbf', C=1, gamma=0.5, random_state=123)
model.fit(X, y)
model.n_support_
model.support_
model.support_vectors_

plot_decision_regions(X, y, model, hide_spines=False)
plt.title('SVM (C=1, gamma=0.5)')

model.score(X, y)

"""
def svm_plot_boundary(model,X,y):
    data = pd.DataFrame(X,columns=['x1','x2'])
    data['y'] = y
    sns.scatterplot(x='x1', y='x2', data=data, s=30, hue=y, palette=['blue','black'])
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.5)
    kernel = model.get_params()['kernel']
    C = model.get_params()['C']
    gamma = model.get_params()['gamma']
    plt.title(f'SVM (kernel = {kernel}, C = {C}, gamma = {gamma})')

svm_plot_boundary(model, X, y)

"""

# Effect of increasing C: overfit

model = SVC(kernel='rbf', C=10000, gamma=0.5, random_state=123)
model.fit(X, y)
plot_decision_regions(X, y, model, hide_spines=False)
plt.title('SVM (C=10000, gamma=0.5)')

# Effect of decreasing C: underfit

model = SVC(kernel='rbf', C=0.01, gamma=0.5, random_state=123)
model.fit(X, y)
plot_decision_regions(X, y, model, hide_spines=False)
plt.title('SVM (C=0.01, gamma=0.5)')

# Effect of increasing gamma: overfit

model = SVC(kernel='rbf', C=1, gamma=50, random_state=123)
model.fit(X, y)
plot_decision_regions(X, y, model, hide_spines=False)
plt.title('SVM (C=1, gamma=50)')

model.score(X, y)

# Effect of decreasing gamma: underfit

model = SVC(kernel='rbf', C=1, gamma=0.05, random_state=123)
model.fit(X, y)
plot_decision_regions(X, y, model, hide_spines=False)
plt.title('SVM (C=1, gamma=0.05)')

model.score(X, y)

# Choose optimal hyperparameters by CV

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(SVC(kernel='rbf', random_state=123), param_grid, cv=kfold)
model.fit(X, y)
model.best_params_

plot_decision_regions(X, y, model, hide_spines=False)
plt.title('Optimal SVM (C=100, gamma=0.1)')

# Evalutae prediction performance

np.random.seed(369)
X_test = np.random.randn(1000, 2)
y_test = np.logical_xor(X_test[:, 0] > 0, X_test[:, 1] > 0)
y_test = np.where(y_test, 1, -1)
model.score(X_test, y_test)

pred = model.predict(X_test)
pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])

## SVM for Binary Classification with the spam data

spam = pd.read_csv('spam.csv')
spam.shape

spam.head(3)

X = spam.iloc[:, :-1]
y = spam.iloc[:, -1]
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=1000, stratify=y, random_state=0)

X.describe()

scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

np.mean(X_train_s, axis=0)
np.std(X_train_s, axis=0)
np.mean(X_test_s, axis=0)
np.std(X_test_s, axis=0)

model = SVC(kernel="linear", random_state=123)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)

model = SVC(kernel="poly", degree=2, random_state=123)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)

model = SVC(kernel="poly", degree=3, random_state=123)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)

model = SVC(kernel="rbf", random_state=123)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)

model = SVC(kernel="sigmoid",random_state=123)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)

param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(SVC(kernel="rbf", random_state=123), param_grid, cv=kfold)
model.fit(X_train_s, y_train)
model.best_params_
model.score(X_test_s, y_test)

pred = model.predict(X_test)
pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])

### SVM for multiple classification with the digits data

digits = load_digits()
dir(digits)

print(digits['DESCR'])

digits.images.shape
digits.images[8]

digits.data.shape

digits.target.shape
pd.Series(digits.target).value_counts()

plt.imshow(digits.images[8], cmap=plt.cm.gray_r)
digits.target[8]

images_8 = digits.images[digits.target==8]

for i in range(1, 10):
    plt.subplot(3, 3, i)
    plt.imshow(images_8[i-1], cmap=plt.cm.gray_r)
plt.tight_layout()

X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)

model = SVC(kernel="linear", random_state=123)
model.fit(X_train, y_train)
model.score(X_test, y_test)

model = SVC(kernel="poly", degree=2, random_state=123)
model.fit(X_train, y_train)
model.score(X_test, y_test)

model = SVC(kernel="poly", degree=3, random_state=123)
model.fit(X_train, y_train)
model.score(X_test, y_test)

model = SVC(kernel='rbf', random_state=123)
model.fit(X_train, y_train)
model.score(X_test, y_test)

model = SVC(kernel="sigmoid",random_state=123)
model.fit(X_train, y_train)
model.score(X_test, y_test)

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1, 10]}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(SVC(kernel='rbf',random_state=123), param_grid, cv=kfold)
model.fit(X_train, y_train)
model.best_params_
model.score(X_test, y_test)

pred = model.predict(X_test)
pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])

plot_confusion_matrix(model, X_test, y_test,cmap='Blues')
plt.tight_layout()

# Support Vector Regression with Boston Housing Data

X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# Radial Kernel

model = SVR(kernel='rbf')
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)

param_grid = {'C': [0.01, 0.1, 1, 10, 50, 100, 150], 'epsilon': [0.01, 0.1, 1, 10], 'gamma': [0.01, 0.1, 1, 10]}
kfold = KFold(n_splits=10, shuffle=True, random_state=1)

model = GridSearchCV(SVR(), param_grid, cv=kfold)
model.fit(X_train_s, y_train)
model.best_params_

model = model.best_estimator_
len(model.support_)
X_train_s.shape
model.support_vectors_
model.score(X_test_s, y_test)

# Comparison with Linear Regression

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)
