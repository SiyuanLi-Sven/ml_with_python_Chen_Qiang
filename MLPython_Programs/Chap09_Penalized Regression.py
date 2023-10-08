### Chapter 9 Penalized Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import lasso_path
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import enet_path

# Decriptive Statistics

prostate = pd.read_csv('prostate.csv')
prostate.shape
prostate.head()

prostate.mean()
prostate.std()


# Ridge regression

X_raw = prostate.iloc[:, :-1]
y = prostate.iloc[:, -1]

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

np.mean(X,axis=0)
np.std(X,axis=0)

model = Ridge()
model.fit(X, y)
model.score(X, y)
model.intercept_
model.coef_

pd.DataFrame(model.coef_, index=X_raw.columns, columns=['Coefficient'])

# Plot ridge coefficient path  , fit_intercept=False

alphas = np.logspace(-3, 6, 100)

pow(10,np.linspace(-3, 6, 100))

coefs = []
for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    coefs.append(model.coef_) 

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('alpha (log scale)')
plt.ylabel('Coefficients')
plt.title('Ridge Cofficient Path')
plt.axhline(0, linestyle='--', linewidth=1, color='k')
plt.legend(X_raw.columns)

# Choose best ridge regularization by LOOCV
# The default CV scoring is to minimize MSE 

model = RidgeCV(alphas=alphas)
model.fit(X, y)
model.alpha_

# Refine the search grid

alphas=np.linspace(1, 10, 1000)
model = RidgeCV(alphas=alphas, store_cv_values=True)
model.fit(X, y)
model.alpha_

model.cv_values_.shape
mse = np.mean(model.cv_values_, axis=0)
np.min(mse)
index_min = np.argmin(mse)
alphas[index_min], mse[index_min]

plt.plot(alphas, mse)
plt.axvline(alphas[index_min], linestyle='--', linewidth=1, color='k')
plt.xlabel('alpha')
plt.ylabel('Mean Squared Error')
plt.title('CV Error for Ridge Regression')
plt.tight_layout()

model.coef_

pd.DataFrame(model.coef_, index=X_raw.columns, columns=['Coefficient'])

# Choose best ridge regularization by 10-fold CV
# Cannot set "random_state" for RidgeCV
# Use R2 as CV scoring

kfold = KFold(n_splits=10, shuffle=True, random_state=1)
model = RidgeCV(alphas=np.linspace(1,10,1000), cv=kfold)
model.fit(X, y)
model.alpha_

### Lasso regression

model = Lasso(alpha=0.1)
model.fit(X, y)
model.score(X, y)

results = pd.DataFrame(model.coef_, index=X_raw.columns, columns=['Coefficient'])
results

model = Lasso().fit(X, y)
model.coef_

Lasso().fit(X, y).coef_

# Plot Lasso coefficient path

# eps=1e-4 means that alpha_min /alpha_max = 1e-4, the smaller eps is the longer is the path
alphas, coefs, _ = lasso_path(X, y, eps=1e-4)
ax = plt.gca()
ax.plot(alphas, coefs.T)
ax.set_xscale('log')
plt.xlabel('alpha (log scale)')
plt.ylabel('Coefficients')
plt.title('Lasso Cofficient Path')
plt.axhline(0, linestyle='--', linewidth=1, color='k')
plt.legend(X_raw.columns)

# Choose best Lasso regularization by 10-fold CV

kfold = KFold(n_splits=10, shuffle=True, random_state=1)
alphas=np.logspace(-4, -2, 100)
model = LassoCV(alphas=alphas, cv=kfold)
model.fit(X, y)
model.alpha_   

pd.DataFrame(model.coef_, index=X_raw.columns, columns=['Coefficient'])

model.mse_path_.shape
mse = np.mean(model.mse_path_, axis=1)

index_min = np.argmin(mse)
alphas[index_min]       # Some errors in model.mse_path_

# CV Error Curve

alphas = np.logspace(-4, -2, 100)
scores = []
for alpha in alphas:
    model = Lasso(alpha=alpha)
    kfold = KFold(n_splits=10, shuffle=True, random_state=1)
    scores_val = -cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
    score = np.mean(scores_val)
    scores.append(score)
    
mse = np.array(scores)
index_min = np.argmin(mse)
alphas[index_min]

plt.plot(alphas, mse)
plt.axvline(alphas[index_min], linestyle='--', linewidth=1, color='k')
plt.xlabel('alpha')
plt.ylabel('Mean Squared Error')
plt.title('CV Error for Lasso')
plt.tight_layout()

# Elastic Net Regression

model = ElasticNet(alpha=0.1, l1_ratio=0.5)
model.fit(X, y)
model.score(X, y)

pd.DataFrame(model.coef_, index=X_raw.columns, columns=['Coefficient'])

# Plot Elastic Net coefficient path

alphas, coefs, _ = enet_path(X, y, eps=1e-4, l1_ratio = 0.5)

ax = plt.gca()
ax.plot(alphas, coefs.T)
ax.set_xscale('log')
plt.xlabel('alpha (log scale)')
plt.ylabel('Coefficients')
plt.title('Elastic Net Cofficient Path (l1_ratio = 0.5)')
plt.axhline(0, linestyle='--', linewidth=1, color='k')
plt.legend(X_raw.columns)

# Choose best ElasticNet hyperparameters by 10-fold CV   
           
alphas = np.logspace(-4, 0, 100)
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
model = ElasticNetCV(cv=kfold, alphas = alphas, l1_ratio=[0.0001, 0.001, 0.01, 0.1, 0.5, 1])
model.fit(X,y)

model.alpha_
model.l1_ratio_
model.score(X,y)

pd.DataFrame(model.coef_, index=X_raw.columns, columns=['Coefficient'])

## Use optimal ridge regression for prediction

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = RidgeCV(alphas=np.linspace(1, 20, 1000))

model.fit(X_train, y_train)

model.alpha_

model.score(X_train, y_train)
model.score(X_test, y_test)


