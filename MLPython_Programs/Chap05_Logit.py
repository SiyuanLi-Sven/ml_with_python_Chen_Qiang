
### Chapter 5 Logit

# Logistic Distribution

import numpy as np
from scipy.stats import logistic
import matplotlib.pyplot as plt
fig,ax = plt.subplots(1, 2, figsize=(8, 4))
x = np.linspace(-5, 5, 100)
ax[0].plot(x, logistic.pdf(x),linewidth=2)
ax[0].vlines(0, 0, .255, linewidth=1)
ax[0].hlines(0, -5, 5, linewidth=1)
ax[0].set_title('Logistic Density')
ax[1].plot(x, logistic.cdf(x), linewidth=2)
ax[1].vlines(0, 0, 1, linewidth=1)
ax[1].hlines(0, -5, 5, linewidth=1)
ax[1].set_title('Logistic CDF')


# Descriptive Statistics for Titanic Data

import pandas as pd
import numpy as np
titanic = pd.read_csv('titanic.csv')

titanic.shape

titanic.head()

freq = titanic.Freq.to_numpy()
index = np.repeat(np.arange(32), freq)
index.shape

titanic = titanic.iloc[index,:]
titanic = titanic.drop('Freq', axis=1)
titanic.head()
titanic.info()
titanic.describe()

pd.crosstab(titanic.Sex, titanic.Survived)
pd.crosstab(titanic.Sex, titanic.Survived, normalize='index')

pd.crosstab(titanic.Age, titanic.Survived, normalize='index')

pd.crosstab(titanic.Class, titanic.Survived, normalize='index')

# Logit estimation

from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from patsy import dmatrices

train, test =  train_test_split(titanic, test_size=0.3, stratify=titanic.Survived, random_state=0)

# Change discrete variable into dummy variables

y_train, X_train = dmatrices('Survived ~ Class + Sex + Age', data=train, return_type='dataframe')

pd.options.display.max_columns = 10
X_train.head()

y_train.head()
y_train = y_train.iloc[:,1]

y_test, X_test = dmatrices('Survived ~ Class + Sex + Age', data=test, return_type='dataframe')
y_test = y_test.iloc[:,1]

model = sm.Logit(y_train, X_train)
results = model.fit()
results.params
np.exp(results.params)   # Odds ratio
results.summary()

margeff = results.get_margeff()
margeff.summary()

# Training error

table = results.pred_table()  # Confusion matrix for training set
table

Accuracy = (table[0, 0] + table[1, 1]) / np.sum(table)
Accuracy

Error_rate = 1 - Accuracy
Error_rate

Sensitivity  = table[1, 1] / (table[1, 0] + table[1, 1])
Sensitivity

Specificity = table[0, 0] / (table[0, 0] + table[0, 1])
Specificity

Recall = table[1, 1] / (table[0, 1] + table[1, 1])
Recall

# Test error

prob = results.predict(X_test)
pred = (prob >= 0.5)
table = pd.crosstab(y_test, pred, colnames=['Predicted'])
table

table = np.array(table)   # Change pandas DataFrame to numpy array

Accuracy = (table[0, 0] + table[1, 1]) / np.sum(table)
Accuracy

Error_rate = 1 - Accuracy
Error_rate

Sensitivity  = table[1, 1] / (table[1, 0] + table[1, 1])
Sensitivity

Specificity = table[0, 0] / (table[0, 0] + table[0, 1])
Specificity

Recall = table[1, 1] / (table[0, 1] + table[1, 1])
Recall

# Use sklearn for Logit

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import plot_roc_curve

model =  LogisticRegression(C=1e10)
model.fit(X_train, y_train)

model.coef_
results.params

model =  LogisticRegression(C=1e10, fit_intercept=False)
model.fit(X_train, y_train)
model.coef_

model.score(X_test, y_test) 

prob = model.predict_proba(X_test)
prob[:5]

pred = model.predict(X_test)
pred[:5]

confusion_matrix(y_test, pred)

   # Accuracy

# For better formating,use pandas
table = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])
table

print(classification_report(y_test, pred, target_names=['Not Survived', 'Survived']))

# ROC and AUC

plot_roc_curve(model, X_test, y_test)
x = np.linspace(0, 1, 100)
plt.plot(x, x, 'k--', linewidth=1)
plt.title('ROC Curve (Test Set)')

cohen_kappa_score(y_test, pred)

