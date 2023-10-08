### Chapter 2 Introduction to Python

## 2.1 Why Use Python

## 2.2 Installation of Python and Spyder

## 2.3 Calculator and Assignment

2 ** 0.5 

pow(2, 0.5)

x = 2 ** 0.5   # x is called a "variable"
x
print(x)

x = 2
type(x)

type(0.2)

type(2.0)

type(2.)

number = 18_000_000_000
number

pow(abs(-2), 0.5)  # a composite function

help('keywords')


## 2.4 Import modules 

# As a general-purpose language, Python's functions are very limited

import numpy as np

np.sqrt(2)

np.log(2)

from numpy import sqrt

sqrt(2)

from numpy import *

log(2)

def mysqrt(x):
    return pow(x, 0.5)

mysqrt(2) 

""" Save the following code as 'hello.py' in the current working directory

print('Hello Python World!')
x = 2
def sqrt(x):
    return pow(x,0.5)
    
"""

import hello as he

he.x

he.sqrt(2)

import sys
sys.path
    
## 2.5 字符串

x = 'This is a string.'
print(x)

type(x)

print("I love Python!")


print("I told me friend,'I love Python!'")


# triple quote
quote = """
To deal with a 14-dimensional space, 
visualizea 3-D space and say "fourteen" 
to yourself very loudly. Everyone does it.  
-- Geoffrey Hinton
"""
print(quote)

name = 'alan turing'
print(name)

name.upper()

name.title()

name = 'Leo Breiman'
name.lower()

name = 'Geoffrey Hinton'
print(f'Hello, {name}!')

print('Hello, {}!'.format(name))

print('Python')

print('\tPython')

print('Languages:\nPython\nR\nStata')

print('Languages:\n\tPython\n\tR\n\tStata')

x = 'C:\new\tom'
print(x)

x = 'C:\\new\\tom'
print(x)

x = r'C:\new\tom'
print(x)

language = 'python '
language.rstrip()
language

language = 'python '
language.rstrip()

language = ' python'
language.lstrip()

language = ' python '
language.strip()

# string as an immutable sequence
x = 'python'
len(x)

x[0]

x[1]

x[-1]

x[-2]

x[1:3]

x[:3]

x[2:]

x[2:-1]

x = 'abcdefg'
x[1:6:2]

# x[0] = 'q'

x = 'Py'
y = 'thon'
print(x + y)

print('Python' + str(3))

print('Hi! ' * 5)

## 2.6 布尔型

2 > 1

2 <= 1

2 == 1

2 != 1

type(2 > 1)

type(True)

type(False)

x = None
type(x)

## 2.7 列表

languages = ['Python', 'R', 'Stata']
print(languages)

type(languages)

languages[0]

languages[-1]

languages[-1] = 'Matlab'
languages

languages.append('Stata')
languages

del languages[-1]
languages

languages.extend(['Stata', 'Eviews'])
languages

languages.insert(2, 'SAS')
languages

last_language = languages.pop()
last_language

languages.remove('SAS')
print(languages)

languages.sort()
languages

languages.sort(reverse=True)
languages

languages = ['Python', 'R', 'Matlab', 'Stata']
sorted(languages)

languages

languages.reverse()
languages

len(languages)

languages = ['Python', 'R', 'Matlab', 'Stata']
languages[1:3]

languages[:3]

languages[1:]

my_languages = ['Python', 'R', 'Stata']
friend_languages = my_languages
friend_languages

friend_languages[-1] = 'C'
friend_languages

my_languages

my_languages = ['Python', 'R', 'Stata']
friend_languages = my_languages[:]
friend_languages

my_languages == friend_languages

my_languages is friend_languages

friend_languages[-1] = 'C'
my_languages

x = 'Python'
x = 'R'

# List in list
languages = [my_languages, friend_languages]
print(languages)

languages[0][1]

# Numerical List
x = [0,1,2,3,4]
print(x)
type(x)

# Mixed list
x = [1, 'Python', True, False, None, ['R', 'Stata']]
print(x)

type(x)

list(range(5))

range(5)

type(range(5))

list(range(1, 11))

x = list(range(1, 11, 2))
print(x)

min(x)

max(x)

sum(x)

## 2.8 元组

dimensions = (150, 4)
dimensions

type(dimensions)

dimensions = 150, 4, 10
dimensions

dimensions[0]

dimensions[1]

dimensions[:2]

# dimensions[1] = 5   # immutable sequence

x = 150; y = 4

x, y = 150, 4
x

y

x, y = y, x
x, y

my_tuple = (3,)
my_tuple

# tuple in list

list0 = [(1, 5), (2, 10), (3, 15)]
list0

list1 = [1, 2, 3]
list2 = [5, 10, 15]
list3 = list(zip(list1, list2))
list3

type(zip(list1, list2))

list4 = list(enumerate(list2))
list4

type(enumerate(list2))

## 2.9 字典

Dict = {'P': 'Python', 'M': 'Matlab', 'S': 'Stata','M': 'Matlab'}
Dict

type(Dict)

Dict['P']
Dict['M']
Dict['S']

Dict.keys()
Dict.values()
Dict.items()

Dict['E'] = 'Eviews'
Dict

Dict['S'] = 'SAS'      # mutable sequence
Dict

del Dict['E']
Dict

# Dictionary in dictionary

computer_languages = {'C': 'C', 'J': 'JavaScript'}
math_languages = {'R': 'R', 'M': 'Matlab'}
languages = {'computer': computer_languages, 'math': math_languages}
languages

languages['math']['M']

# List in dictionary

computer_languages = ['C', 'JavaScript']
math_languages = ['R', 'Matlab']
languages = {'computer': computer_languages, 'math': math_languages}
languages

# Dictionary in list

computer_languages = {'C': 'C', 'J': 'JavaScript'}
math_languages = {'R': 'R', 'M': 'Matlab'}
languages = [computer_languages, math_languages]
languages

## 2.10 集合

my_languages = {'Python', 'R', 'Stata'}
my_languages

type(my_languages)

'Stata' in my_languages

'Stata' not in my_languages

friend_languages = {'Python', 'R', 'C'}

our_languages = my_languages | friend_languages
our_languages

common_languages = my_languages & friend_languages
common_languages

languages = ['Python', 'R', 'Matlab', 'Stata','R','Stata']

unique_languages = set(languages)
unique_languages

## 2.11 数组

import numpy as np

arr = np.arange(5)
arr

type(arr)

arr.ndim

arr.shape

arr.dtype

arr * 2

arr + 10

np.sqrt(arr)

np.exp(arr)

type(np.sqrt)

arr = np.arange(10)
arr = arr.reshape(5, 2)
arr

arr.shape

arr.ndim

arr.reshape(2, 5)

list1 = [3, 5, 2, 1]
arr1 = np.array(list1)
arr1

list2 = [[3, 5, 2, 1], [6, 7, 9, 8]]
arr2 = np.array(list2)
arr2

arr2.ndim

arr2.shape

np.zeros(10)

np.zeros((2, 5))

arr3 = np.ones((2, 3, 2))
arr3
arr3.shape

arr4 = np.array([1, 2, 3], dtype=np.float64)
arr4.dtype

arr5 = arr4.astype(np.float32)
arr5.dtype

arr = np.arange(1, 7).reshape(2, 3)
arr

arr = np.arange(1, 7).reshape(2, -1)
arr

arr + arr

arr * arr

arr = np.arange(10)
arr
arr[5]
arr[5:8]

arr[5:8] = 12   # broadcasting
arr

arr_slice = arr[5:8]
arr_slice

arr_slice[1] = 123
arr_slice
arr

arr = np.arange(10)
arr[5:8] = 12
arr

arr_slice = arr[5:8].copy()
arr_slice

arr_slice[1] = 123
arr

arr = np.arange(1, 10).reshape(3, 3)
arr
arr[0]
arr[0][1]
arr[:2]
arr[:2, 1:]

arr[:, 1:]
arr[:2, :]


arr[arr > 4]

select = arr > 4
select

arr[select]

arr[[1, 0, 2]]   # fancy indexing

np.where(arr < 5, 0, 1)


np.sum(arr)
arr.sum()

np.mean(arr)
arr.mean()

arr.mean(axis=0)
arr.mean(axis=1)

arr.cumsum()

(arr > 5).sum()

arr > 5 

arr = np.array([3, 2, 3, 1, 1])
arr.sort()
arr

np.unique(arr)


# Linear Algebra

arr = np.arange(1, 10).reshape(3, 3)
arr
arr.T

# Element-wise multiplication
arr * arr.T

# matrix multiplication
np.dot(arr, arr.T)  
arr.dot(arr.T)
arr @ arr.T

from numpy.linalg import inv, eig

inv(arr)

eigenvalues, eigenvectors = eig(arr)
eigenvalues
eigenvectors

## 2.12 数据框

import pandas as pd

# Series
s = pd.Series([3, 2, 6, 1, 5])
s

type(s)

s.index
s.values
s1 = s.sort_values()
s1

s[0]
s[1]
s[[0,2]]

s.index = ['a', 'b', 'c', 'd', 'e']
s

s['a']
s[['a', 'c']]

s = pd.Series([3, 1, 1, 2, 2, 2])
s

s.unique()
s.value_counts()

# Data Frame

Dict = {'student ID': [1, 2, 3, 4], 'age': [18, 17, 18, 19],
        'sex': ['female', 'male', 'male', 'female'], 'score': [90, 85, 85, 80]}
df = pd.DataFrame(Dict)
df
type(df)

df.index
df.columns
df.values

df1 = df.sort_values(by='score')
df1

df['age']
type(df['age'])

df.age

df[['age', 'sex']]
type(df[['age', 'sex']])

df.loc[0,'age']

df.iloc[0, 1]

df.iloc[1, 2:]

df.loc[:,['sex', 'score']]

df.iloc[:, 2:]

df.loc[df.score > 80, :]

df.score.value_counts()

df.sex.value_counts()

pd.crosstab(df.sex, df.score)


arr = np.arange(12).reshape(3, 4)
arr
df2 = pd.DataFrame(arr, columns=['x1', 'x2', 'x3', 'x4'])
df2

df2.columns = ['y1', 'y2', 'y3', 'y4']
df2

df2['y5'] = [5, 7, 9]
df2

df2['y6'] = np.array([5, 3, 1])
df2

df2 = df2.drop('y6', axis=1)
df2

df2 = df2.drop('y5', axis='columns')
df2

# 2.13 缺失值

x = pd.Series([1, None, 3, 5])
x

len(x)

x.sum()
x.mean()
np.mean(x)

x = pd.Series([1, np.nan, 3, 5])
x

x.isna()
x.isnull()

x.notna()
x.notnull()

x.isna().sum()


## 2.14 描述性统计

import pandas as pd
import seaborn as sns
iris = sns.load_dataset('iris')

iris.info()

iris.head()
iris.tail()

iris.describe()
iris.species.value_counts()

iris.var()

iris.corr()

iris_grouped = iris.groupby('species')

type(iris_grouped)

iris_grouped.corr()

iris_grouped.describe()

pd.options.display.max_columns = 50   # default to 100 rows and 20 columns

iris_grouped.describe()

iris_grouped.petal_length.describe()

iris_grouped.mean()

iris_grouped.agg(['mean', 'std'])

iris_grouped.agg({'sepal_length': 'mean', 'sepal_width': 'mean', 'petal_length': 'std', 'petal_width': 'std'})

def iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)

iris_grouped.agg(iqr)

iris_grouped.apply(iqr)

def zscore(x):
    return (x - x.mean()) / x.std()

transformed = iris_grouped.transform(zscore)

transformed.head(3)

transformed.groupby(iris.species).agg(['mean', 'std'])


## 2.15 用matplotlib画图

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
x.shape

plt.plot(x,np.sin(x), 'k-')
plt.plot(x,np.cos(x), 'b--')
plt.show()

# Matlab-style interface

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x, np.sin(x))
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x))

# Object-oriented interface

fig, ax = plt.subplots(2)
type(fig)
type(ax)
ax
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))

plt.figure()
plt.plot(x, np.sin(x))
plt.axhline(0, linewidth=1)
plt.axvline(0, linewidth=1)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('A Sine Curve')

plt.figure()
plt.plot(x,np.sin(x), label='sin(x)', color='k')
plt.plot(x,np.cos(x), label='cos(x)', color='b')
plt.legend(loc='lower left')

## 2.16 用pandas与seaborn画图

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

iris = sns.load_dataset('iris')
iris.head(3)

# Histogram
iris.sepal_length.hist()

iris.sepal_length.hist(bins=20, color='b')
plt.xlabel('sepal_length')
plt.title('Iris Data')

iris.hist()

# Density plot
iris.sepal_length.plot.density()
iris.sepal_length.plot.kde()

iris.plot.density()
iris.plot.density(subplots=True)

sns.distplot(iris.sepal_length, rug=True)

# Box plot
iris.sepal_width.plot.box()
iris.plot.box()
sns.boxplot(x='species', y='sepal_width', data=iris)

# Scatterplot
sns.scatterplot(x='petal_length', y='petal_width', data=iris)
sns.scatterplot(x='petal_length', y='petal_width', data=iris, hue='species')
sns.scatterplot(x='petal_length', y='petal_width', data=iris, style='species', color='k')
sns.scatterplot(x='petal_length', y='petal_width', data=iris, style='species', hue='species')

# Scatter Matrix
sns.pairplot(data=iris, height=2)
sns.pairplot(iris, vars=['sepal_length', 'sepal_width', 'petal_length'], diag_kind='kde')
sns.pairplot(data=iris, height=2, hue='species')

# Joint Plot
sns.jointplot(x="sepal_length", y="sepal_width", data=iris, kind="kde")

# Save figure
sns.jointplot(x="sepal_length", y="sepal_width", data=iris, kind="kde")
plt.savefig('bivariate_density.jpg')

sns.jointplot(x="sepal_length", y="sepal_width", data=iris, kind="kde")
plt.savefig('bivariate_density', format='png', dpi=300)

## 2.17 读写数据

import pandas as pd
import seaborn as sns
iris = sns.load_dataset('iris')

iris.to_csv('iris.csv', index=False)

iris1 = pd.read_csv('iris.csv')

iris1.head()

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris2 = pd.read_csv(url, header=None)

iris2.head()

iris2.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris2

iris.to_excel('iris.xlsx', index=False)
iris3 = pd.read_excel('iris.xlsx')
iris3.head()

iris.to_json('iris.json')
pd.read_json('iris.json')

iris.to_stata('iris.dta', write_index=False)
pd.read_stata('iris.dta')

## 2.18 随机抽样

import random

random.random()

random.random()

random.seed(0)
random.random()

random.seed(1)
random.choice([1, 2, 3, 4, 5])

random.seed(2)
random.choices([1, 2, 3, 4, 5], k=3)   # with replacement

random.seed(10)
random.choices(['head', 'tail'], k=3)  # with replacement

random.seed(20)
random.sample([1, 2, 3, 4, 5], k=3)    # without replacement

import numpy as np
np.random.seed(1)
np.random.rand(5)

np.random.seed(1)
np.random.randn(5)

np.random.seed(1)
np.random.normal(1, 2, 5)   # sample from N(1,2^2)

np.random.seed(1)
x = np.random.randn(100).reshape(50, 2)
data = pd.DataFrame(x, columns=['x1', 'x2'])
data.head()

import seaborn as sns
sns.scatterplot(x='x1', y='x2', data=data)

## 2.19 条件语句

# Example 1
x = 5
if x > 0:
    x = 1
print(x)

# Example 2
x = 5
if x > 0:
    print('Original x is positive.')
    x = 1
    print(f'Updated x is {x}.')


# Example 3
x = -2
if x > 0:
    x = 1
else:
    x = -1
print(x)

# Example 4
x = 0
if x > 0:
    x = 1
elif x < 0:
    x = -1
else:
    x = 0
print(x)

# Example 5
x = 5
y= 10
if x > 0:
    if y > 0:
        z = x * y
print(z)

# Example 6
x = 5
if x > 0: x = 1
print(x)


x = (111,
     222)
print(x)

y = [111,
     222]
print(y)

z = {'a':111,
     'b':222}
print(z)

w = (111+
     222)
print(w)

# Vectorize operation
import numpy as np
x = np.arange(-3, 4)
print(x)
np.where(x > 0 , 1, -1)

a, b = 10, 20
if a < b:
    min = a
else:
    min = b
min

min = a if a < b else b
min


## 2.20 循环语句

for i in range(5):
    print('Hello!')

for n in [0, 1, 1, 2, 3, 5]:
    print(n ** 2)

import matplotlib.pyplot as plt
x = np.linspace(0, 1, 100)
plt.plot(x, x, 'k')
for i in range(2, 11):
    plt.plot(x, x ** i, 'k')

i = 1
while i < 6:
    print('Hello!')
    i = i + 1

n = 1000000
x = np.random.rand(n)
y = np.random.rand(n)
z = np.zeros(n)

%timeit z = x + y

%%timeit 
for i in range(len(z)):
    z[i] = x[i] + y[i]

%timeit for i in range(len(z)): z[i] = x[i] + y[i]    # still use line magic function

import time

start_time = time.perf_counter()
z = x + y
time.perf_counter() - start_time

start_time = time.perf_counter()
for i in range(len(z)):
    z[i] = x[i] + y[i]
time.perf_counter() - start_time

# List comprehension

list1 = list(range(5))
list1

list2 =[]
for i in list1:
    list2.append(i ** 2)
list2

list2 = [i ** 2 for i in list1]
list2

%timeit for i in list1: list2.append(i ** 2)

%timeit list2 = [i ** 2 for i in list1]

list3 = [i ** 2 for i in list1 if i > 2]
list3

# Example: The Number Guessing Game

from random import randint
x = randint(0, 100)
tries = 0
print("I'm thinking of a number between 1 and 100.")

while True:
   tries += 1
   guess = int(input("Have a guess :"))
   if guess > x:
       print("Too high...")
   elif guess < x:
      print("Too low...")
   else:
       break
print(f'You are right! The number was {x}, and you only tried {tries} times!')


## 2.21 函数

def f(x):
    return x + 1

f(2)

type(f)

f?

def f(x):
    """
    Add 1 to a number   
    """
    return x + 1

g = f

?g

help(g)

g??

import numpy as np
np.mean?
help(np.mean)

def f(x1, x2, x3):
    print(x1, x2, x3)
    
f(1, 2, 3)

f(x3=3, x1=1, x2=2)

f(1, x3=3, x2=2)

def f(x1, x2=2, x3=3):
    print(x1, x2, x3)

f(1)

f(x1=1)

f(1, 3, 5)


def f(*args): print(args)

f(1)

f(1, 2, 3, 4)

def f(**args): print(args)

f(x1=1,x2=2)

def f(x, *pargs, **kargs): print(x, pargs, kargs)

f(1, 2, 3, y=4, z=5)

# Factory Function

def maker(n):
    def power(x):
        return x ** n
    return power

square = maker(2)
square(3)

cube = maker(3)
cube(3)

# Recursive Function

def fib(n):
    if n == 0 or n == 1:
        return n
    else:
        return fib(n - 1) + fib(n - 2)
fib(3)
fib(6)

# Example 1 of function scope
x= 66
def f(y):
    z = x + y 
    return z
f(10)

# Example 2 of function scope
x= 66 
def f(y):
    x = 88
    z = x + y 
    return z
f(10)
x

# Example 3 of function scope
x= 66 
def f(y):
    global x
    x = 88
    z = x + y 
    return z
f(10)
x

import builtins

print(dir(builtins))


lambda x: x + 1

(lambda x: x + 1)(2)

add_one = lambda x: x + 1

add_one(2)

def square(x):
    return x ** 2
list1 = [1, 2, 3]
list(map(square, list1))

list(map(lambda x: x ** 2, list1))


## 2.22 类

class Employee:
    def __init__(self, name, job='staff', pay=5000):
        self.name = name
        self.job = job
        self.pay = pay

mary = Employee('Mary Taylor')
mary.name
mary.job
mary.pay

john = Employee('John E. Smith', job='teacher', pay=10000)
john.name
john.job
john.pay

john.name.split()
john.name.split()[0]
john.name.split()[-1]


class Employee:
    def __init__(self,name, job='staff', pay=5000):
        self.name = name
        self.job = job
        self.pay = pay
    def firstName(self):
        return self.name.split()[0]
    def lastName(self):
        return self.name.split()[-1]
    def giveRaise(self,percent):
        self.pay = round(self.pay * (1 + percent),2)

john = Employee('John E. Smith', job='teacher', pay=10000)
john.name
john.job
john.pay

john.firstName()
john.lastName()
john.giveRaise(0.1)
john.pay

print(john)

class Employee:
    def __init__(self, name, job='staff', pay=5000):
        self.name = name
        self.job = job
        self.pay = pay
    def firstName(self):
        return self.name.split()[0]
    def lastName(self):
        return self.name.split()[-1]
    def giveRaise(self,percent):
        self.pay = round(self.pay * (1 + percent), 2)
    def __repr__(self):
        return f'Employee: {self.name}. Job: {self.job}. Pay: {self.pay}.'

john = Employee('John E. Smith', job='teacher', pay=10000)
print(john)


class Manager(Employee):
    """
    Manager as a subclass of Employee
    """
    def giveRaise(self, percent, bonus=0.1):
        Employee.giveRaise(self, percent + bonus)
      
Manager?
        
tom = Manager('Tom Watson', 'manager', 20000)
tom.name
tom.job
tom.pay

tom.firstName()
tom.lastName()
tom.giveRaise(0.1)
tom.pay

print(tom)

print(dir(tom))
tom.__class__
tom.__dict__
tom.__format__


## 2.23 进一步学习Python的资源







