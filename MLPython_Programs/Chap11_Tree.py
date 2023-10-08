### Chapter 11 Decision Trees

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor,export_text
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_boston
from sklearn.metrics import cohen_kappa_score

# Data Preparation

Boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(Boston.data, Boston.target, test_size=0.3, random_state=0)

# Regression Tree 

model = DecisionTreeRegressor(max_depth=2, random_state=123)
model.fit(X_train, y_train)
model.score(X_test, y_test)

print(export_text(model,feature_names=list(Boston.feature_names)))

plot_tree(model, feature_names=Boston.feature_names, node_ids=True, rounded=True, precision=2)

# Graph total impurities versus ccp_alphas 

model = DecisionTreeRegressor(random_state=123)
path = model.cost_complexity_pruning_path(X_train, y_train)

plt.plot(path.ccp_alphas, path.impurities, marker='o', drawstyle='steps-post')
plt.xlabel('alpha (cost-complexity parameter)')
plt.ylabel('Total Leaf MSE')
plt.title('Total Leaf MSE vs alpha for Training Set')

max(path.ccp_alphas),  max(path.impurities)

# Choose optimal ccp_alpha via CV

param_grid = {'ccp_alpha': path.ccp_alphas} 
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(DecisionTreeRegressor(random_state=123), param_grid, cv=kfold)
model.fit(X_train, y_train)

model.best_params_
model = model.best_estimator_
model.score(X_test,y_test)

plot_tree(model, feature_names=Boston.feature_names, node_ids=True, rounded=True, precision=2)

model.get_depth()
model.get_n_leaves()
model.get_params()

# Visualize Feature Importance
          
model.feature_importances_

sorted_index = model.feature_importances_.argsort()
sorted_index 

X = pd.DataFrame(Boston.data, columns=Boston.feature_names)

plt.barh(range(X.shape[1]), model.feature_importances_[sorted_index])
plt.yticks(np.arange(X.shape[1]), X.columns[sorted_index])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Decision Tree')
plt.tight_layout()

# Visualize prediction fit

pred = model.predict(X_test)

plt.scatter(pred, y_test, alpha=0.6)
w = np.linspace(min(pred), max(pred), 100)
plt.plot(w, w)
plt.xlabel('pred')
plt.ylabel('y_test')
plt.title('Tree Prediction')

from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)
model.score(X_test, y_test)


## Classification Tree with bank dataset

bank = pd.read_csv('bank-additional.csv', sep=';')

bank.shape

pd.options.display.max_columns = 70 
bank.head()

# Drop 'duration' variable
bank = bank.drop('duration', axis=1)

bank.y.value_counts()
bank.y.value_counts(normalize=True)

bank.groupby(['y']).mean()

X_raw = bank.iloc[:, :-1]
X = pd.get_dummies(X_raw)
X.head(2)

y = bank.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=1000, random_state=1)

# Classification Tree 

model = DecisionTreeClassifier(max_depth=2, random_state=123)
model.fit(X_train, y_train)
model.score(X_test, y_test)
plot_tree(model, feature_names=X.columns, node_ids=True, rounded=True, precision=2)

# Graph total impurities versus ccp_alphas 

model = DecisionTreeClassifier(random_state=123)
path = model.cost_complexity_pruning_path(X_train, y_train)

plt.plot(path.ccp_alphas, path.impurities, marker='o', drawstyle='steps-post')
plt.xlabel('alpha (cost-complexity parameter)')
plt.ylabel('Total Leaf Impurities')
plt.title('Total Leaf Impurities vs alpha for Training Set')

max(path.ccp_alphas),  max(path.impurities)

# Choose optimal ccp_alpha via CV

param_grid = {'ccp_alpha': path.ccp_alphas}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(DecisionTreeClassifier(random_state=123), param_grid, cv=kfold)
model.fit(X_train, y_train)     

model.best_params_

model = model.best_estimator_
model.score(X_test, y_test)

plot_tree(model, feature_names=X.columns, node_ids=True, impurity=True, proportion=True, rounded=True, precision=2)

# Feature importance

model.feature_importances_

sorted_index = model.feature_importances_.argsort()
plt.barh(range(X_train.shape[1]), model.feature_importances_[sorted_index])
plt.yticks(np.arange(X_train.shape[1]), X_train.columns[sorted_index])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Decision Tree')
plt.tight_layout()

# Prediction Performance 
     
pred = model.predict(X_test)
table = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])
table

table = np.array(table)
Accuracy = (table[0, 0] + table[1, 1]) / np.sum(table)
Accuracy

Sensitivity  = table[1, 1] / (table[1, 0] + table[1, 1])
Sensitivity

cohen_kappa_score(y_test, pred)

# Use a different threshold for prediction

prob = model.predict_proba(X_test)
prob
model.classes_

prob_yes = prob[:, 1]
pred_new = (prob_yes >= 0.1)

table = pd.crosstab(y_test, pred_new, rownames=['Actual'], colnames=['Predicted'])
table

table = np.array(table)
Accuracy = (table[0, 0] + table[1, 1]) / np.sum(table)
Accuracy

Sensitivity  = table[1, 1] / (table[1, 0] + table[1, 1])
Sensitivity

## Entropy criterion

param_grid = {'ccp_alpha': path.ccp_alphas}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(DecisionTreeClassifier(criterion='entropy', random_state=123), param_grid, cv=kfold)

model.fit(X_train, y_train)     
model.score(X_test, y_test)

pred = model.predict(X_test)
pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])

## Decision boundary for iris data

from sklearn.datasets import load_iris
from mlxtend.plotting import plot_decision_regions

X,y = load_iris(return_X_y=True)
X2 = X[:, 2:4]

model = DecisionTreeClassifier(random_state=123)
path = model.cost_complexity_pruning_path(X2, y)
param_grid = {'ccp_alpha': path.ccp_alphas}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(DecisionTreeClassifier(random_state=123), param_grid, cv=kfold)
model.fit(X2, y)
model.score(X2, y)

plot_decision_regions(X2, y, model)
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.title('Decision Boundary for Decision Tree')
