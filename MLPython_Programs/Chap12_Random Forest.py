### Chapter 12 Random Forest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris, load_boston
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import plot_roc_curve
from sklearn.inspection import plot_partial_dependence
from mlxtend.plotting import plot_decision_regions


# Motorcycle Example: Tree vs. Bagging

mcycle = pd.read_csv('mcycle.csv')
mcycle.head()

X = np.array(mcycle.times).reshape(-1, 1)
y = mcycle.accel 

# Single tree estimation  best_estimator_.

model = DecisionTreeRegressor(random_state=123)
path = model.cost_complexity_pruning_path(X, y)
param_grid = {'ccp_alpha': path.ccp_alphas}
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(DecisionTreeRegressor(random_state=123), param_grid, cv=kfold)
pred_tree = model.fit(X, y).predict(X)

sns.scatterplot(x='times', y='accel', data=mcycle, alpha=0.6)
plt.plot(X, pred_tree, 'b')
plt.title('Single Tree Estimation')

# Bagging estimation

model = BaggingRegressor(base_estimator=DecisionTreeRegressor(random_state=123), n_estimators=500, random_state=0)
pred_bag = model.fit(X, y).predict(X)

sns.scatterplot(x='times', y='accel', data=mcycle, alpha=0.6)
plt.plot(X, pred_bag, 'b')
plt.title('Bagging Estimation')

# Alternatively,one could use 'RandomForestRegressor', which by default 
# sets max_features = n_features that is de facto bagging. The results are slightly different.
# The advantage of 'BaggingRegressor' is the option to use different base learners.

"""
model = RandomForestRegressor(n_estimators=500, random_state=0)
model.fit(X, y)
model.score(X,y)
pred = model.predict(X)
sns.scatterplot(x='times',y='accel',data=mcycle,alpha=0.6)
plt.plot(X,pred,'b')
plt.title('Bagging Estimation')
"""

# Bagging for Regression on Boston Housing Data

Boston = load_boston()
X = pd.DataFrame(Boston.data, columns=Boston.feature_names)
y = Boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = BaggingRegressor(base_estimator=DecisionTreeRegressor(random_state=123), n_estimators=500, oob_score=True, random_state=0)

model.fit(X_train, y_train)
pred_oob = model.oob_prediction_
mean_squared_error(y_train, pred_oob)
model.oob_score_
model.score(X_test, y_test)

# Comparison with OLS

model = LinearRegression().fit(X_train, y_train)
model.score(X_test, y_test)

# OOB Errors

oob_errors = []
for n_estimators in range(100,301):
    model = BaggingRegressor(base_estimator=DecisionTreeRegressor(random_state=123),
                          n_estimators=n_estimators, n_jobs=-1, oob_score=True, random_state=0)
    model.fit(X_train, y_train)
    pred_oob = model.oob_prediction_
    oob_errors.append(mean_squared_error(y_train, pred_oob))

plt.plot(range(100, 301), oob_errors)
plt.xlabel('Number of Trees')
plt.ylabel('OOB MSE')
plt.title('Bagging OOB Errors')

# Random Forest for Regression on Boston Housing Data

max_features=int(X_train.shape[1] / 3)
max_features

model = RandomForestRegressor(n_estimators=500, max_features=max_features, random_state=0)
model.fit(X_train, y_train)
model.score(X_test, y_test)

# Visualize prediction fit

pred = model.predict(X_test)

plt.scatter(pred, y_test, alpha=0.6)
w = np.linspace(min(pred), max(pred), 100)
plt.plot(w, w)
plt.xlabel('pred')
plt.ylabel('y_test')
plt.title('Random Forest Prediction')

# Feature Importance Plot

model.feature_importances_
sorted_index = model.feature_importances_.argsort()

plt.barh(range(X.shape[1]), model.feature_importances_[sorted_index])
plt.yticks(np.arange(X.shape[1]), X.columns[sorted_index])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Random Forest')
plt.tight_layout()

# Partial Dependence Plot, features, feature_names = ['LSTAT', 'RM'], n_jobs=-1

plot_partial_dependence(model, X, ['LSTAT', 'RM'], n_jobs=-1)

"""
# Partial Dependence Plot in 3D

fig = plt.figure()
features = ('LSTAT','RM')
pdp, axes = partial_dependence(model, X, features=features)
XX, YY = np.meshgrid(axes[0], axes[1])
Z = pdp[0].T
ax = Axes3D(fig)
surf = ax.plot_surface(XX, YY, Z, rstride=5, cstride=5,
                       cmap=plt.cm.BuPu, edgecolor='k')
ax.set_xlabel(features[0])
ax.set_ylabel(features[1])
ax.set_zlabel('Partial dependence')
#  pretty init view
ax.view_init(elev=18, azim=40)
plt.colorbar(surf)
plt.suptitle('Partial dependence of house value on\n'
             'LSTAT and RM, with Random Forest')
plt.subplots_adjust(top=0.9)
"""

# Choose best max_features parameter via Test set

scores = []
for max_features in range(1, X.shape[1] + 1):
    model = RandomForestRegressor(max_features=max_features,
                                  n_estimators=500, random_state=123)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores.append(score)

index = np.argmax(scores)
range(1, X.shape[1] + 1)[index]

plt.plot(range(1, X.shape[1] + 1), scores, 'o-')
plt.axvline(range(1, X.shape[1] + 1)[index], linestyle='--', color='k', linewidth=1)
plt.xlabel('max_features')
plt.ylabel('R2')
plt.title('Choose max_features via Test Set')

print(scores)

## Comparison of Test Errors for Single Tree, Bagging and Random Forest

# Random Forest

scores_rf = []
for n_estimators in range(1, 301):
    model = RandomForestRegressor(max_features=9,
                                  n_estimators=n_estimators, random_state=123)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    scores_rf.append(mse)

# Bagging

scores_bag = []
for n_estimators in range(1, 301):
    model = BaggingRegressor(base_estimator=DecisionTreeRegressor(random_state=123), n_estimators=n_estimators, random_state=0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    scores_bag.append(mse)

# Single Tree

model = DecisionTreeRegressor()
path = model.cost_complexity_pruning_path(X_train, y_train)
param_grid = {'ccp_alpha': path.ccp_alphas}
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(DecisionTreeRegressor(random_state=123), param_grid, cv=kfold, scoring='neg_mean_squared_error')
model.fit(X_train, y_train)
score_tree = -model.score(X_test, y_test)
scores_tree = [score_tree for i in range(1, 301)]

plt.plot(range(1, 301), scores_tree, 'k--', label='Single Tree')
plt.plot(range(1, 301), scores_bag, 'k-', label='Bagging')
plt.plot(range(1, 301), scores_rf, 'b-', label='Random Forest')
plt.xlabel('Number of Trees')
plt.ylabel('MSE')
plt.title('Test Error')
plt.legend()

# Choose best hyperparameters via CV

max_features = range(1, X.shape[1] + 1)                   
param_grid = {'max_features': max_features }
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(RandomForestRegressor(n_estimators=300, random_state=123), 
                     param_grid, cv=kfold, scoring='neg_mean_squared_error', return_train_score=True)

model.fit(X_train, y_train)
model.best_params_
cv_mse = -model.cv_results_['mean_test_score']

plt.plot(max_features, cv_mse, 'o-')
plt.axvline(max_features[np.argmin(cv_mse)], linestyle='--', color='k', linewidth=1)
plt.xlabel('max_features')
plt.ylabel('MSE')
plt.title('CV Error for Random Forest')

### Random Forest for Classification 

# Descriptive Statistics for Sonar dataset

Sonar = pd.read_csv('Sonar.csv')
Sonar.shape
Sonar.head(2)
Sonar.Class.value_counts()

X = Sonar.iloc[:, :-1]
y = Sonar.iloc[:, -1]

sns.heatmap(X.corr(), cmap='Blues')
plt.title('Correlation Matrix')
plt.tight_layout()

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=50, random_state=1)

# Single Tree  as benchmark

model = DecisionTreeClassifier()
path = model.cost_complexity_pruning_path(X_train, y_train)
param_grid = {'ccp_alpha': path.ccp_alphas}
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(DecisionTreeClassifier(random_state=123), param_grid, cv=kfold)
model.fit(X_train, y_train)   
model.score(X_test, y_test)

# Logistic Regression

model = LogisticRegression(C=1e10, max_iter=500)
model.fit(X_train, y_train)   
model.score(X_test, y_test)

# Random Forest

model = RandomForestClassifier(n_estimators=500, max_features='sqrt', random_state=123)
model.fit(X_train, y_train)
model.score(X_test, y_test)

# Choose optimal mtry parameter via CV

y_train_dummy = pd.get_dummies(y_train)
y_train_dummy = y_train_dummy.iloc[:, 1]
             
param_grid = {'max_features': range(1, 11) }
kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=1)
model = GridSearchCV(RandomForestClassifier(n_estimators=300, random_state=123), param_grid, cv=kfold)

model.fit(X_train, y_train_dummy)
model.best_params_

model = RandomForestClassifier(n_estimators=500, max_features=8, random_state=123)
model.fit(X_train, y_train)
model.score(X_test, y_test)

# Feature Importance Plot

model.feature_importances_

sorted_index = model.feature_importances_.argsort()
plt.barh(range(X.shape[1]), model.feature_importances_[sorted_index])
plt.yticks(np.arange(X.shape[1]), X.columns[sorted_index])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Random Forest')

# Partial Dependence Plot

plot_partial_dependence(model, X_train, ['V11'])

# Prediction Performance 

pred = model.predict(X_test)
table = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])
table

table = np.array(table)
Accuracy = (table[0, 0] + table[1, 1]) / np.sum(table)
Accuracy

Sensitivity  = table[1 , 1] / (table[1, 0] + table[1, 1])
Sensitivity

Specificity = table[0, 0] / (table[0, 0] + table[0, 1])
Specificity

Recall = table[1, 1] / (table[0, 1] + table[1, 1])
Recall

cohen_kappa_score(y_test, pred)

plot_roc_curve(model, X_test, y_test)
x = np.linspace(0, 1, 100)
plt.plot(x, x, 'k--', linewidth=1)
plt.title('ROC Curve for Random Forest')

## Iris example: Decision Boundary for Random Forest 

X,y = load_iris(return_X_y=True)
X2 = X[:, 2:4]

model = RandomForestClassifier(n_estimators=500, max_features=1, random_state=1)
model.fit(X2,y)
model.score(X2,y)

plot_decision_regions(X2, y, model)
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.title('Decision Boundary for Random Forest')
