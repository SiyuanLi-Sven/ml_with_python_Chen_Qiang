### Chapter 6 Multinomial Logit

# Decriptive Statistics

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score

Glass = pd.read_csv('Glass.csv')
Glass.shape

Glass.head()

Glass.Type.value_counts()

sns.distplot(Glass.Type, kde=False)

sns.boxplot(x='Type', y='Mg', data=Glass, palette="Blues")

# Multinomial Logit    

X = Glass.iloc[:,:-1]
y = Glass.iloc[:,-1]

X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.3, stratify=y, random_state=0)

model =  LogisticRegression(multi_class='multinomial',
                            solver = 'newton-cg', C=1e10, max_iter=1e3)
model.fit(X_train, y_train)

model.n_iter_
model.intercept_
model.coef_
model.score(X_test, y_test)    # Accuracy

prob = model.predict_proba(X_test)
prob[:3]

pred = model.predict(X_test)
pred[:5]

table = confusion_matrix(y_test, pred)
table

# For better display,use pandas 
table = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])
table

sns.heatmap(table,cmap='Blues', annot=True)
plt.tight_layout()

print(classification_report(y_test, pred))

cohen_kappa_score(y_test, pred)

