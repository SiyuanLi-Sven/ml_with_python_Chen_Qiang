### Chapter 7 Discriminant Analysis

# Decriptive Statistics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score

iris = load_iris()
dir(iris)
iris.feature_names
iris.feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

X = pd.DataFrame(iris.data, columns=iris.feature_names)
X.corr()
sns.heatmap(X.corr(), cmap='Blues', annot=True)

y = iris.target

# LDA for full sample

model = LinearDiscriminantAnalysis()
model.fit(X, y)
model.score(X, y)

model.priors_
model.means_

model.explained_variance_ratio_

model.scalings_

lda_loadings = pd.DataFrame(model.scalings_, index=iris.feature_names, columns=['LD1', 'LD2'])
lda_loadings

lda_scores = model.fit(X, y).transform(X)
lda_scores.shape
lda_scores[:5, :]

lda_scores = model.fit_transform(X, y)

LDA_scores = pd.DataFrame(lda_scores, columns=['LD1', 'LD2'])
LDA_scores['Species'] = iris.target
LDA_scores.head()

d = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
LDA_scores['Species'] = LDA_scores['Species'].map(d) 
LDA_scores.head()

sns.scatterplot(x='LD1', y='LD2', data=LDA_scores, hue='Species')

# Plot decision boundary for LDA with two features

X2 = X.iloc[:, 2:4]

model = LinearDiscriminantAnalysis()
model.fit(X2, y)
model.score(X2, y)
model.explained_variance_ratio_

# pip install mlxtend (machine learning extension library), or to avoid timeout
# pip --default-timeout=100 install mlxtend 

from mlxtend.plotting import plot_decision_regions

plot_decision_regions(np.array(X2), y, model)
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.title('Decision Boundary for LDA')

# LDA for split sample

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.3, stratify=y, random_state=123)

model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)
model.score(X_test, y_test)    # Accuracy

prob = model.predict_proba(X_test)
prob[:3]

pred = model.predict(X_test)
pred[:5]

confusion_matrix(y_test, pred)

print(classification_report(y_test, pred))

cohen_kappa_score(y_test, pred)

# QDA for split sample

model = QuadraticDiscriminantAnalysis()
model.fit(X_train, y_train)
model.score(X_test, y_test)    # Accuracy

prob = model.predict_proba(X_test)
prob[:3]

pred = model.predict(X_test)
pred[:5]

confusion_matrix(y_test, pred)

print(classification_report(y_test, pred))

cohen_kappa_score(y_test, pred)

# Plot decision boundary for QDA with two features

X2 = X.iloc[:, 2:4]
model = QuadraticDiscriminantAnalysis()
model.fit(X2, y)
model.score(X2, y)

plot_decision_regions(np.array(X2), y, model)
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.title('Decision Boundary for QDA')

