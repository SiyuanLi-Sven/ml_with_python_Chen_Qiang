### Chapter 4 Linear Regression

## Engle(1857)'s Data 

import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns

data = sm.datasets.engel.load_pandas().data
data.shape
data.head()

model = smf.ols('foodexp ~ income', data=data)
results = model.fit()
results.params
results.summary()
sns.regplot(x='income', y='foodexp', data=data)

## Cobb Douglas Production Function

import pandas as pd
data = pd.read_csv('cobb_douglas.csv')
data.shape
data.head()

model = smf.ols('lny ~ lnk + lnl', data=data)
results = model.fit()
results.params

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

xx = np.linspace(data.lnk.min(), data.lnk.max(), 100)
yy = np.linspace(data.lnl.min(), data.lnl.max(), 100)
xx.shape, yy.shape
XX, YY = np.meshgrid(xx,yy)
XX.shape, YY.shape
ZZ = results.params[0] + XX * results.params[1] +  YY * results.params[2] 

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(data.lnk, data.lnl, data.lny,c='b')
ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3, cmap='viridis')
ax.set_xlabel('lnk')
ax.set_ylabel('lnl')
ax.set_zlabel('lny')
plt.margins(0)


## Simulated Data to Demonstrate Overfitting

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# True DGP and Scatterplot

np.random.seed(42)
x = np.random.rand(10)
y = np.sin(2 * np.pi * x)+np.random.normal(0, 0.3, 10)
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
w = np.linspace(0, 1, 100)
plt.plot(w, np.sin(2 * np.pi * w))

# Regression Plots 

fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, subplot_kw=dict(yticks=[]))
fig.subplots_adjust(hspace=0.1, wspace=0.1)
for m in range(1, 10):
  ax = fig.add_subplot(3, 3, m)
  ax.set_xlim([0, 1])
  ax.set_ylim([-1.5, 1.5])
  ax.set_yticks(np.arange(-1.5, 2, 0.5))
  ax.plot(w, np.sin(2 * np.pi * w), 'k', linewidth=1)
  sns.regplot(x, y, order=m, ci=None, scatter_kws={'s':15}, line_kws={'linewidth':1})
  ax.text(0.6, 1, 'M = '+ str(m))
plt.tight_layout()

# Plot of Training Set R2

R2 = []   # initialize R2 as an empty list 
train_error = []    # initialize training error
for m in range(1, 9):    
    stats = np.polyfit(x, y, m, full=True)
    ssr = stats[1]
    train_error.append((ssr / 10) ** 0.5)
    r2 = 1 - ssr/(sum((y-y.mean()) ** 2))
    R2.append(r2)
R2.append(1)
train_error.append(0)    

plt.plot(range(1, 10), R2, 'o-')
plt.xlabel('Degree of Polynomial')
plt.ylabel('R2')
plt.title('In-sample Fit')

# Plot of Training and Test Errors

test_error = []    # initialize training error
for m in range(1, 10):    
    coefs = np.polyfit(x, y, m)
    np.random.seed(123)
    x_new = np.random.rand(100)
    y_new = np.sin(2 * np.pi * x_new) + np.random.normal(0, 0.3, 100)
    pred = np.polyval(coefs, x_new)    
    ssr = (sum((pred - y_new) ** 2) / 100) ** 0.5
    test_error.append(ssr)

plt.plot(range(1, 10), train_error, 'o--k', label='Training Error')
plt.plot(range(1, 10), test_error, 'o-b', label='Test Error')
plt.ylim(-0.05, 1)
plt.xlabel('Degree of Polynomial')
plt.ylabel('Root Mean Squared Errors')
plt.title('Training Error vs. Test Error')
plt.legend()

print(test_error)
    
## Example of Boston Housing Dataset Using statsmodel

import pandas as pd 
from sklearn.datasets import load_boston
import statsmodels.formula.api as smf

dataset = load_boston()
type(dataset)

dir(dataset)

Boston = pd.DataFrame(dataset.data, columns=dataset.feature_names)

Boston['MEDV'] = dataset.target

Boston.info()

pd.options.display.max_columns = 20 

Boston.head()

Boston.describe()

# Regression formulae

model = smf.ols('MEDV ~ RM', data=Boston)
results = model.fit()
results.summary()     # Use print(results.summary()) to remove triple quotes

results = smf.ols('MEDV ~ RM + AGE', data=Boston).fit()
print(results.summary().tables[1])

results = smf.ols('MEDV ~ RM + AGE + RM:AGE', data=Boston).fit()
results.params

results = smf.ols('MEDV ~ RM * AGE', data=Boston).fit()
results.params

results = smf.ols('MEDV ~ RM * AGE + I(RM**2) + I(AGE**2)', data=Boston).fit()
results.params

all_columns = "+".join(dataset.feature_names)
all_columns
formula = 'MEDV~' + all_columns
formula
results = smf.ols(formula, data=Boston).fit()
print(results.summary().tables[1])

formula_NoAGE = formula + "-AGE"
formula_NoAGE

results = smf.ols(formula_NoAGE, data=Boston).fit()
results.params

import statsmodels.api as sm

X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = dataset['target']
model = sm.OLS(y, X)
results = model.fit()
results.params

# Validation Set Approach

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=0)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

model = LinearRegression()
model.fit(X_train, y_train)
model.coef_
model.score(X_test, y_test)

pred = model.predict(X_test)
pred.shape
mean_squared_error(y_test, pred)
r2_score(y_test, pred)

# Use a different random seed to compare test MSE

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=12345)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
mean_squared_error(y_test, pred)
r2_score(y_test, pred)

# Cross Validation

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RepeatedKFold


X, y = load_boston(return_X_y = True)

model = LinearRegression()

kfold = KFold(n_splits=10,shuffle=True, random_state=1)
scores = cross_val_score(model, X, y, cv=kfold)
scores
scores.mean()
scores.std()

scores_mse = -cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
scores_mse
scores_mse.mean()

# Use a different seed for 10-fold CV

kfold = KFold(n_splits=10, shuffle=True, random_state=123)
scores_mse = -cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
scores_mse.mean()

# 10-fold CV Repeated 10 times

rkfold = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)
scores_mse = -cross_val_score(model, X, y, cv=rkfold, scoring='neg_mean_squared_error')

scores_mse.shape
scores_mse.mean()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(pd.DataFrame(scores_mse))
plt.xlabel('MSE')
plt.title('10-fold CV Repeated 10 Times')  

# LOOCV

loo = LeaveOneOut()
scores_mse = -cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error')
scores_mse.mean()  
