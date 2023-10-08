### Chapter 18 Python for Data Science 

import numpy as np
import pandas as pd

## Read file

myfile = open('myfile.txt', 'w')
type(myfile)

myfile.write('Stay Hungry.\n')
myfile.write('Stay Foolish.\n')
myfile.close()

with open('myfile.txt', 'w') as myfile:
    myfile.write('Stay Hungry.\n')
    myfile.write('Stay Foolish.\n')

myfile = open('myfile.txt','r')
myfile.readline()
myfile.readline()
myfile.readline()

myfile = open('myfile.txt')
myfile.readlines()

open('myfile.txt').read()

print(open('myfile.txt').read())

for line in open('myfile.txt'):
    print(line)

for line in open('myfile.txt'):
    print(line, end='')

for line in open('myfile.txt'):
    print(line.rstrip())
    
with open('myfile.txt', 'a') as myfile:
    myfile.write('-- Steve Jobs\n')
 
print(open('myfile.txt').read())

# An Example of Pi Digits

with open('pi_digits.txt', 'w') as file_object:
    file_object.write('3.1415926535\n')
    file_object.write('8979323846\n')
    file_object.write('2643383279\n')

print(open('pi_digits.txt').read())

pi_string = ''
for line in open('pi_digits.txt'):
    pi_string += line.strip()

print(pi_string)
len(pi_string)


## Read text data

df = pd.read_csv('ex1.csv')
df

df = pd.read_table('ex1.csv', sep=',')
df

pd.read_csv('ex2.csv', header=None)

pd.read_csv('ex2.csv', names = ['a', 'b', 'c', 'd', 'message'])


list(open('ex3.txt'))
df = pd.read_table('ex3.txt', sep='\s+')
df

list(open('ex4.csv'))
pd.read_csv('ex4.csv', skiprows=[0, 2, 3])

list(open('ex5.csv'))
result = pd.read_csv('ex5.csv')
result
pd.isna(result)

pd.read_csv('ex5.csv', na_values=['foo'])

sentinels = {'something': 'two', 'message': 'foo'}
pd.read_csv('ex5.csv', na_values=sentinels)

pd.read_csv('ex6.csv')

pd.read_csv('ex6.csv', nrows=5)

chunks = pd.read_csv('ex6.csv', chunksize=2000)
type(chunks)

means = []; mins = []; maxs = []
for chunk in chunks:
    means.append(chunk.one.mean())
    mins.append(chunk.one.min())
    maxs.append(chunk.one.max())
means 
mins
maxs
np.mean(means), np.max(mins), np.max(maxs)

## Missing Data

from numpy import nan as NA

df = pd.DataFrame([[1, NA, 3, 5], [7, 2, NA, NA], [NA, NA, NA, NA], [9, 4, 6, 3]],
                  columns = ['x1', 'x2', 'x3', 'x4'])
df

df.dropna()

df.dropna(how='all')

df['x4'] = NA
df
df.dropna(axis=1, how='all')

df.dropna(thresh=2)

df.fillna(0)

df.fillna({'x1': 1, 'x2': 1, 'x3': 1, 'x4': 0})

df.fillna({'x1': 1, 'x2': 1, 'x3': 1, 'x4': 0}, inplace=True)
df

np.random.seed(1)
matrix = np.random.randn(6, 3)
df = pd.DataFrame(matrix)
df

df.iloc[2:5, 1] = NA
df.iloc[4, 2] = NA
df

df.fillna(method='ffill')

df.fillna(method='ffill', limit=2)

df.fillna(method='bfill')

df.fillna(df.mean())

df.fillna(df.median())

df.interpolate()

## Example: Titanic data from Kaggle

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.shape, test.shape
pd.options.display.max_columns = 15
train.head()

train.dtypes.sort_values()

train.isna().sum()
test.isna().sum()

# Embarked
mode_Embarked = train.Embarked.mode()
mode_Embarked 
type(mode_Embarked)

train.Embarked.fillna(mode_Embarked[0], inplace=True)
# titanic.Embarked.mode() returns a pandas Series, and we want its 0th element

# Fare

test.groupby('Pclass').mean().Fare
test.Fare.fillna(test.groupby('Pclass').transform('mean').Fare, inplace=True)

# Cabin
train.Cabin.fillna('unknown', inplace=True)
test.Cabin.fillna('unknown', inplace=True)

# Age 

train['Name'].head()

train.loc[0,'Name'].split('.')[0].split(',')[1]

train.loc[0,'Name'].split('.')[0].split(',')[1].strip()

train['Title'] = train.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())

train[['Name','Title']].head()

test['Title'] = test.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())

train['Title'].value_counts()

test['Title'].value_counts()


newtitles={
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"}

train['Title'] = train.Title.map(newtitles)
test['Title'] = test.Title.map(newtitles)

train['Title'].value_counts()
test['Title'].value_counts()

train.groupby(['Title','Sex', 'Pclass']).median().Age

train.Age.fillna(train.groupby(['Title','Sex', 'Pclass']).transform('median').Age, inplace=True)

test.Age.fillna(test.groupby(['Title','Sex', 'Pclass']).transform('median').Age, inplace=True)

train.isna().sum()
test.isna().sum()


## Duplicate Data

data = pd.DataFrame({'k1': ['one', 'two'] * 3 + ['two'], 'k2': [1, 1, 2, 3, 3, 4, 4]})
data

data.duplicated()

data.drop_duplicates()

data.drop_duplicates(keep='last')

data.drop_duplicates(['k1', 'k2'])


## Merge Data 

np.random.seed(1)
matrix1 = np.random.randn(2, 2)
matrix1

np.random.seed(12)
matrix2 = np.random.randn(2, 2)
matrix2

np.concatenate((matrix1, matrix2))
np.vstack((matrix1, matrix2))

np.concatenate((matrix1, matrix2), axis=1)
np.hstack((matrix1, matrix2))

df1 = pd.DataFrame(matrix1)
df2 = pd.DataFrame(matrix2)
pd.concat([df1, df2])

pd.concat([df1, df2], axis=1)

df1.append(df2)

df3 = pd.DataFrame({'A': ['A1','A2'], 'B': ['B1','B2'], 'C': ['C1', 'C2']})
df3

df4 = pd.DataFrame({'B': ['B1','B2'], 'C': ['C1', 'C2'], 'D': ['D1','D2'], })
df4

pd.concat([df3, df4])
pd.concat([df3, df4], join='inner')


df5 = pd.DataFrame({'employee': ['Bob', 'John', 'Lisa', 'Sue'], 'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df5

df6 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Tom', 'Sue'], 'hire_date': [2004, 2008, 2012, 2014]})
df6

pd.merge(df5, df6, on='employee')

pd.merge(df5, df6, on='employee', how='outer')

pd.merge(df5, df6, on='employee', how='left')

pd.merge(df5, df6, on='employee', how='right')

df7 = pd.DataFrame({'name': ['Lisa', 'Bob', 'Tom', 'Sue'], 'hire_date': [2004, 2008, 2012, 2014]})
df7

pd.merge(df5, df7, left_on='employee', right_on='name')

pd.merge(df5, df7, left_on='employee', right_on='name').drop('name', axis=1)


df8 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'var1': range(7)})
df8

df9 = pd.DataFrame({'key': ['a', 'b', 'd'], 'var2': range(3)})
df9

pd.merge(df8, df9, on='key')

df5a = df5.set_index('employee')
df5a

df6a = df6.set_index('employee')
df6a

pd.merge(df5a, df6a, left_index=True, right_index=True)

df5a.join(df6a)

pd.merge(df5a, df6, left_index=True, right_on='employee')

df10 = pd.DataFrame({'employee': ['Bob', 'John', 'Lisa', 'Sue'], 'rank': [1, 2, 3, 4]})
df10

df11 = pd.DataFrame({'employee': ['Bob', 'John', 'Lisa', 'Sue'], 'rank': [3, 1, 4, 2]})
df11

pd.merge(df10, df11, on='employee')

pd.merge(df10, df11, on='employee', suffixes=['_L', '_R'])

## Pipeline in Sci-kit Learn

from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

svm = SVC()
svm.fit(X_train_s, y_train)
svm.score(X_test_s, y_test)

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(SVC(), param_grid=param_grid, cv=kfold)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)

model.best_params_


# Building pipeline

pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC())])
type(pipe)

pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)

# Using pipeline in grid search

param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100], 'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
model = GridSearchCV(pipe, param_grid=param_grid, cv=kfold)
model.fit(X_train, y_train)
model.score(X_test, y_test)

model.best_params_

# Accessing step attributes

pipe.named_steps['svm'].classes_
pipe.named_steps['svm'].n_support_
pipe.named_steps['svm'].support_
pipe.named_steps['svm'].support_vectors_

# Convenient pipeline creation with make_pipeline

pipe_long = Pipeline([("scaler", StandardScaler()), ("svm", SVC())])
pipe_short = make_pipeline(StandardScaler(), SVC())
pipe_long.steps
pipe_short.steps

