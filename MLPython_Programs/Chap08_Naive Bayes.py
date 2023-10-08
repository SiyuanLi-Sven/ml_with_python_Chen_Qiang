### Chapter 8 Naive Bayes

# Decriptive Statistics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

spam = pd.read_csv('spam.csv')
spam.shape

pd.options.display.max_columns = 60 
spam.head(1)

spam.spam.value_counts()
spam.spam.value_counts(normalize=True)

spam.iloc[:, :5].plot.hist(subplots=True, bins=100)

spam.iloc[:, -4:].plot.hist(subplots=True, bins=100)

X = spam.iloc[:, :-1]
y = spam.iloc[:, -1]
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

model = GaussianNB()
model.fit(X_train, y_train)
model.score(X_test, y_test) 

model = MultinomialNB(alpha=0)
model.fit(X_train, y_train)
model.score(X_test, y_test) 

model = MultinomialNB(alpha=1)   # Laplacian Correction
model.fit(X_train, y_train)
model.score(X_test, y_test) 

model = ComplementNB(alpha=1)
model.fit(X_train, y_train)
model.score(X_test, y_test) 

model = BernoulliNB(alpha=1)
model.fit(X_train, y_train)
model.score(X_test, y_test) 

model = BernoulliNB(binarize=0.1, alpha=1)
model.fit(X_train, y_train)
model.score(X_test, y_test) 

# A naive approach to choose hyperparameter by grid search

best_score = 0
for binarize in np.arange(0, 1.1, 0.1):
    for alpha in np.arange(0, 1.1, 0.1):
        model = BernoulliNB(binarize=binarize, alpha=alpha)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test) 
        if score > best_score:
            best_score = score
            best_parameters = {'binarize': binarize, 'alpha': alpha}
best_score
best_parameters      

# Choose hyperparameters by training-validation-test sets

X_trainval, X_test, y_trainval, y_test =  train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_train, X_val, y_train, y_val =  train_test_split(X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=123)

y_train.shape, y_val.shape, y_test.shape

best_val_score = 0
for binarize in np.arange(0, 1.1, 0.1):
    for alpha in np.arange(0, 1.1, 0.1):
        model = BernoulliNB(binarize=binarize, alpha=alpha)
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val) 
        if score > best_val_score:
            best_val_score = score
            best_val_parameters = {'binarize': binarize, 'alpha': alpha}
best_val_score
best_val_parameters

model = BernoulliNB(**best_val_parameters)
model.fit(X_trainval, y_trainval)
model.score(X_test, y_test) 
   
# Choose hyperparameters by 10-fold CV

best_score = 0
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
for binarize in np.arange(0, 1.1, 0.1):
    for alpha in np.arange(0, 1.1, 0.1):
        model = BernoulliNB(binarize=binarize, alpha=alpha)        
        scores = cross_val_score(model, X_trainval, y_trainval, cv=kfold) 
        score = np.mean(scores)
        if score > best_score:
            best_score = score
            best_parameters = {'binarize': binarize, 'alpha': alpha}
best_score
best_parameters

model = BernoulliNB(**best_parameters)
model.fit(X_trainval, y_trainval)
model.score(X_test, y_test) 

# Faster Implementation with GridSearchCV

param_grid = {'binarize': np.arange(0, 1.1, 0.1), 'alpha': np.arange(0, 1.1, 0.1)}

model = GridSearchCV(BernoulliNB(), param_grid, cv=kfold)

model.fit(X_trainval, y_trainval)

model.score(X_test, y_test)
model.best_params_
model.best_score_       # best cross validation accuracy
model.best_estimator_

# Visualize CV results

results = pd.DataFrame(model.cv_results_)
results.head(2)

scores = np.array(results.mean_test_score).reshape(11,11)
ax = sns.heatmap(scores, cmap='Blues', annot=True, fmt='.3f')
ax.set_xlabel('binarize')
ax.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
ax.set_ylabel('alpha')
ax.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.tight_layout()

# Prediction and Evaludation

prob = model.predict_proba(X_test)
prob[:3]

pred = model.predict(X_test)
pred[:3]

table = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])
table

table = np.array(table)
Accuracy = (table[0, 0] + table[1, 1]) / np.sum(table)
Accuracy

Error_rate = 1 - Accuracy
Error_rate

Sensitivity  = table[1, 1]/(table[1, 0] + table[1, 1])
Sensitivity

Specificity = table[0, 0] / (table[0, 0] + table[0, 1])
Specificity

Recall = table[1, 1] / (table[0, 1] + table[1, 1])
Recall

plot_roc_curve(model, X_test, y_test)
x = np.linspace(0, 1, 100)
plt.plot(x, x,'k--', linewidth=1)
plt.title('ROC Curve for Bernoulli Naive Bayes')

cohen_kappa_score(y_test, pred)

# Decision boundary for Naive Bayes on iris data

from sklearn.datasets import load_iris
from mlxtend.plotting import plot_decision_regions

X, y = load_iris(return_X_y=True)
X2 = X[:, 2:4]

model = GaussianNB()
model.fit(X2, y)
model.score(X2, y)

plot_decision_regions(X2, y, model)
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.title('Decision Boundary for Gaussian Naive Bayes')
