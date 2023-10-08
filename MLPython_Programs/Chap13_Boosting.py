### Chapter 13 Boosting 

## Simulated Example

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor

np.random.seed(1)
x = np.random.uniform(0, 1, 500)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, 500)
data = pd.DataFrame(x, columns=['x'])
data['y'] = y
w = np.linspace(0, 1, 100)

# Plot the DGP

sns.scatterplot(x='x', y='y', s=20, data=data, alpha=0.3)
plt.plot(w, np.sin(2 * np.pi * w))
plt.title('Data Generating Process')

# Plot GBM with different numbers of trees

for i, m in zip([1, 2, 3, 4], [1, 10, 100, 1000]):
    model = GradientBoostingRegressor(n_estimators=m, max_depth=1, learning_rate=1, random_state=123)
    model.fit(x.reshape(-1, 1), y)
    pred = model.predict(w.reshape(-1, 1))
    plt.subplot(2, 2, i)
    plt.plot(w, np.sin(2 * np.pi * w), 'k', linewidth=1)
    plt.plot(w, pred, 'b')
    plt.text(0.65, 0.8, f'M = {m}')
plt.subplots_adjust(wspace=0.4, hspace=0.4)

## GBM for Regression on Boston Housing Data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import plot_roc_curve
from sklearn.inspection import plot_partial_dependence


Boston = load_boston()
X = pd.DataFrame(Boston.data, columns=Boston.feature_names)
y = Boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = GradientBoostingRegressor(random_state=123)
model.fit(X_train, y_train)
model.score(X_test, y_test)

# Comparison with OLS

LinearRegression().fit(X_train, y_train).score(X_test, y_test)


# Choose best hyperparamters by RandomizedSearchCV

param_distributions = {'n_estimators': range(1, 300), 'max_depth': range(1, 10),
                       'subsample': np.linspace(0.1,1,10), 'learning_rate': np.linspace(0.1, 1, 10)}
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
model = RandomizedSearchCV(estimator=GradientBoostingRegressor(random_state=123),
                           param_distributions=param_distributions, cv=kfold, n_iter=100, random_state=0)
model.fit(X_train, y_train)
model.best_params_
model = model.best_estimator_
model.score(X_test, y_test)

# Feature Importance Plot

model.feature_importances_

sorted_index = model.feature_importances_.argsort()
plt.barh(range(X.shape[1]), model.feature_importances_[sorted_index])
plt.yticks(np.arange(X.shape[1]), X.columns[sorted_index])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Gradient Boosting')
plt.tight_layout()

# Partial Dependence Plot

plot_partial_dependence(model, X_train, ['LSTAT', 'RM'])

# Number of trees and prediction performance

scores = []
for n_estimators in range(1, 301):
    model = GradientBoostingRegressor(n_estimators=n_estimators, subsample=0.5, max_depth=5, learning_rate=0.1, random_state=123)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    scores.append(mse)

index = np.argmin(scores)
range(1, 301)[index]

plt.plot(range(1, 301), scores)
plt.axvline(range(1, 301)[index], linestyle='--', color='k', linewidth=1)
plt.xlabel('Number of Trees')
plt.ylabel('MSE')
plt.title('MSE on Test Set')

## Boosting for Binary Classification 

spam = pd.read_csv('spam.csv')
spam.shape
spam.head(3)

X = spam.iloc[:, :-1]
y = spam.iloc[:, -1]
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

# AdaBoost 

model = AdaBoostClassifier(random_state=123)
model.fit(X_train, y_train)
model.score(X_test, y_test)

# Gradient Boosting

model = GradientBoostingClassifier(random_state=123)
model.fit(X_train, y_train)
model.score(X_test, y_test)

# Choose best hyperparameters by RandomizedSearchCV

param_distributions = {'n_estimators': range(1, 300), 'max_depth': range(1, 10),
                       'subsample': np.linspace(0.1,1,10), 'learning_rate': np.linspace(0.1, 1, 10)}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = RandomizedSearchCV(estimator=GradientBoostingClassifier(random_state=123),
              param_distributions=param_distributions, n_iter=10, cv=kfold, random_state=66)
model.fit(X_train, y_train)
model.best_params_
model = model.best_estimator_
model.score(X_test, y_test)

# Feature Importance Plot

model.feature_importances_

sorted_index = model.feature_importances_.argsort()
plt.barh(range(X.shape[1]), model.feature_importances_[sorted_index])
plt.yticks(np.arange(X.shape[1]), X.columns[sorted_index])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Gradient Boosting')

# Partial Dependence Plot

plot_partial_dependence(model, X_train, ['A.52', 'A.53'])

# Prediction Performance 

prob = model.predict_proba(X_test)

pred = model.predict(X_test)
table = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])
table

table = np.array(table)
Accuracy = (table[0, 0] + table[1, 1]) / np.sum(table)
Accuracy

Sensitivity  = table[1, 1] / (table[1, 0] + table[1, 1])
Sensitivity

Specificity = table[0, 0] / (table[0, 0] + table[0, 1])
Specificity

Recall = table[1, 1] / (table[0, 1] + table[1, 1])
Recall

cohen_kappa_score(y_test, pred)

plot_roc_curve(model, X_test, y_test)
x = np.linspace(0, 1, 100)
plt.plot(x, x, 'k--', linewidth=1)

### Boosting for Multiple Classification 

Glass = pd.read_csv('Glass.csv')
Glass.shape

Glass.head()
Glass.Type.value_counts()

X = Glass.iloc[:, :-1]
y = Glass.iloc[:, -1]
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

model = GradientBoostingClassifier(random_state=123)
model.fit(X_train, y_train)
model.score(X_test, y_test)

# Choose best hyperparameters by RandomizedSearchCV

param_distributions = {'n_estimators': range(1, 300),'max_depth': range(1, 10),
                       'subsample': np.linspace(0.1, 1, 10),'learning_rate': np.linspace(0.1, 1, 10)}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
model = RandomizedSearchCV(estimator=GradientBoostingClassifier(random_state=123),
              param_distributions=param_distributions, cv=kfold, random_state=66)
model.fit(X_train, y_train)
model.best_params_
model = model.best_estimator_
model.score(X_test, y_test)

sorted_index = model.feature_importances_.argsort()
plt.barh(range(X.shape[1]), model.feature_importances_[sorted_index])
plt.yticks(np.arange(X.shape[1]), X.columns[sorted_index])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Gradient Boosting')

# Partial Dependence Plot

plot_partial_dependence(model, X_train, ['Mg'], target=1)
plt.title('Reponse: Glass.Type = 1')

# Prediction Performance

prob = model.predict_proba(X_test)
prob[:3, :]

pred = model.predict(X_test)
pred[:5]

pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])

### XGBoost for Boston Housing Data

# conda install py-xgboost
import xgboost as xgb

X, y= load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300, max_depth=6, 
         subsample=0.6, colsample_bytree=0.8, learning_rate=0.1, random_state=0)
model.fit(X_train, y_train)
model.score(X_test, y_test)

pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
rmse

# Compute RMSE using CV

params = {'objective': 'reg:squarederror', 'max_depth': 6, 'subsample': 0.6,
          'colsample_bytree': 0.8, 'learning_rate': 0.1}

dtrain = xgb.DMatrix(data=X_train, label=y_train)
type(dtrain)

results = xgb.cv(dtrain=dtrain, params=params, nfold=10, metrics="rmse",
                    num_boost_round=300, as_pandas=True, seed=123)
results.shape
results.tail()

# Plot CV Errors
plt.plot(range(1, 301), results['train-rmse-mean'], 'k', label='Training Error')
plt.plot(range(1, 301), results['test-rmse-mean'], 'b', label='Test Error')
plt.xlabel('Number of Trees')
plt.ylabel('RMSE')
plt.axhline(0, linestyle='--', color='k', linewidth=1)
plt.legend()
plt.title('CV Errors for XGBoost')

### XGBoost for spam Data

spam = pd.read_csv('spam.csv')
X = spam.iloc[:, :-1]
y = spam.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=300, max_depth=6, 
         subsample=0.6, colsample_bytree=0.8, learning_rate=0.1, random_state=0)
model.fit(X_train, y_train)
model.score(X_test, y_test)

prob = model.predict_proba(X_test)
prob[:5, :]

pred = model.predict(X_test)
pred[:5]

pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])


