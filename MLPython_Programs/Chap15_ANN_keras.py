### Chapter 15  Artificial Neural Network 

"""
Step 1. In Anaconda Prompt
conda create -n tensorflow_env tensorflow
conda activate tensorflow_env
conda install spyder-kernels
conda install keras
conda install scikit-learn
conda install pandas
conda install matplotlib
conda install seaborn
python -c "import sys; print(sys.executable)"

Step 2. 
# copy the path returned by that command 
# it should end in python, pythonw, python.exe or pythonw.exe, depending on your operating system

Step 3. Start Spyder, navigate to Preferences > Python Interpreter > Use the following interpreter, 
and paste the path from Step 3 into the text box. Then restart Spyder. 

Step 4. Use tensorflow

Step 5. When you are done with tensorflow, you can deactivate the tensorflow environment, and go back to the normal environement
 
conda deactivate 

Step 6. To use tensorflow in the future, in Anaconda Prompt

conda activate tensorflow_env

"""

import numpy as np
import os
import random as rn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import regularizers
import keras
from keras.datasets import mnist, reuters, boston_housing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical

## Regression problem with Boston Housing Data

(X_train, y_train), (X_test, y_test) = boston_housing.load_data(test_split=0.2, seed=113)

X_train.shape
X_test.shape
y_train[:5]
y_test[:5]

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test) 

def set_my_seed():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(1)
    rn.seed(12345)
    tf.random.set_seed(123)
    
set_my_seed()

def build_model():
    model = Sequential() 
    model.add(Dense(units=256, activation='relu', input_shape=(X_train_s.shape[1],)))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=1))
    model.compile(optimizer='rmsprop',loss='mse', metrics=['mse'])
    return model

model = build_model()
model.summary()

hist = model.fit(X_train_s, y_train, validation_split=0.25, epochs=300, batch_size=16, shuffle=False)

type(hist)
Dict = hist.history
Dict.keys()

val_mse = Dict['val_mse']
min(val_mse)

index = np.argmin(val_mse)
index

plt.plot(Dict['mse'], 'k', label='Train')
plt.plot(Dict['val_mse'], 'b', label='Validation')
plt.axvline(index,linestyle='--', color='k')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.title('Mean Squared Error')
plt.legend()

train_mse = Dict['mse']
min(train_mse)

index_train = np.argmin(train_mse)
index_train

set_my_seed()
model = build_model()
model.fit(X_train_s, y_train, epochs=index+1, batch_size=16, verbose=0)

model.evaluate(X_test_s, y_test)
pred = model.predict(X_test_s)
pred.shape

pred = np.squeeze(pred)
pred.shape

np.corrcoef(y_test, pred) ** 2


## Binary Classification with Spam Data

Spam = pd.read_csv('spam.csv')
Spam.shape
Spam.head()

X = Spam.iloc[:, :-1]
y = Spam.iloc[:, -1]
y = pd.get_dummies(y).iloc[:, 1]

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, stratify=y, test_size=1000, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, stratify=y_trainval, test_size=1000, random_state=321)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_val_s = scaler.transform(X_val)
X_trainval_s = scaler.transform(X_trainval)
X_test_s = scaler.transform(X_test)

set_my_seed()
def build_model():
    model = Sequential() 
    model.add(Dense(units=256, activation='relu', input_shape=(X_train_s.shape[1],)))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
model.summary()
hist = model.fit(X_train_s, y_train, validation_data=(X_val_s, y_val), epochs=50, batch_size=64, shuffle=False)

hist.history.keys()

val_loss = hist.history['val_loss']
index_min = np.argmin(val_loss)
index_min

plt.plot(hist.history['loss'], 'k', label='Training Loss')
plt.plot(val_loss, 'b', label='Validation Loss')
plt.axvline(index_min, linestyle='--',color='k')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

val_accuracy = hist.history['val_accuracy']
np.max(val_accuracy)

index_max = np.argmax(val_accuracy)
index_max

plt.plot(hist.history['accuracy'], 'k', label='Training Accuracy')
plt.plot(val_accuracy, 'b', label='Validation Accuracy')
plt.axvline(index_max, linestyle='--', color='k')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


set_my_seed()
model = build_model()
hist = model.fit(X_trainval_s, y_trainval, epochs=index_max+1, batch_size=64, verbose=0, shuffle=False)
test_loss, test_accuracy = model.evaluate(X_test_s, y_test)
test_loss, test_accuracy 

prob = model.predict(X_test_s)
prob[:5]

pred = model.predict_classes(X_test_s)
pred = np.squeeze(pred)
pred[:5]

pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])

# Weight Decay

set_my_seed()
def build_model():
    model = Sequential() 
    model.add(Dense(units=256, activation='relu', kernel_regularizer=regularizers.l2(0.0001), input_shape=(X_train_s.shape[1],)))
    model.add(Dense(units=256, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
hist = model.fit(X_train_s, y_train, validation_data=(X_val_s, y_val), verbose=0, epochs=50, batch_size=64, shuffle=False)
val_accuracy = hist.history['val_accuracy']
index = np.argmax(val_accuracy)
index

set_my_seed()
model = build_model()
hist = model.fit(X_trainval_s, y_trainval, epochs=index+1, batch_size=64, verbose=0, shuffle=False)
test_loss, test_accuracy = model.evaluate(X_test_s, y_test)
test_loss, test_accuracy 

# Dropout

set_my_seed()
def build_model():
    model = Sequential() 
    model.add(Dense(units=256, activation='relu', input_shape=(X_train_s.shape[1],)))
    model.add(Dropout(0.20))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.20))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
hist = model.fit(X_train_s, y_train, validation_data=(X_val_s, y_val), verbose=0, epochs=50, batch_size=64, shuffle=False)
val_accuracy = hist.history['val_accuracy']
index = np.argmax(val_accuracy)
index

set_my_seed()
model = build_model()
hist = model.fit(X_trainval_s, y_trainval, epochs=index+1, batch_size=64, verbose=0, shuffle=False)
test_loss, test_accuracy = model.evaluate(X_test_s, y_test)
test_loss, test_accuracy 


## Multiple Classification with Reuters Data

(X_trainval, y_trainval_original), (X_test, y_test_original) = reuters.load_data(num_words=1000)

X_trainval.shape

print(X_trainval[0])

y_trainval_original[0]

Y_trainval_original = pd.DataFrame(y_trainval_original, columns=['topic'])
Y_trainval_original.hist(bins=46)

X_test.shape

def vectorize_lists(lists, dimension=1000):
    results = np.zeros((len(lists), dimension))
    for i, list in enumerate(lists):
        results[i, list] = 1
    return results

X_trainval = vectorize_lists(X_trainval)
X_test = vectorize_lists(X_test)

X_trainval.shape
X_test.shape

y_trainval = to_categorical(y_trainval_original)
y_test = to_categorical(y_test_original)

y_trainval.shape, y_test.shape

X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, stratify=y_trainval_original, test_size=1000, random_state=321)

set_my_seed()
def build_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.25))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(46, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
model.summary()

hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=64, shuffle=False)

hist.history.keys()

val_loss = hist.history['val_loss']
index_min = np.argmin(val_loss)
index_min

plt.plot(hist.history['loss'], 'k', label='Training Loss')
plt.plot(val_loss, 'b', label='Validation Loss')
plt.axvline(index_min, linestyle='--', color='k')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

val_accuracy = hist.history['val_accuracy']
np.max(val_accuracy)

index_max = np.argmax(val_accuracy)
index_max

plt.plot(hist.history['accuracy'], 'k', label='Training Accuracy')
plt.plot(val_accuracy, 'b', label='Validation Accuracy')
plt.axvline(index_max, linestyle='--', color='k')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

set_my_seed()
model = build_model()
model.summary()
model.fit(X_trainval, y_trainval, epochs=index_max+1, batch_size=64, shuffle=False)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
test_loss, test_accuracy 

prob = model.predict(X_test)
prob[0].shape
prob[0]

pred = model.predict_classes(X_test)
pred[:5]

table = confusion_matrix(y_test_original, pred)
table
table.shape
sns.heatmap(table, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()


## CNN with MNIST Data

# Use only test data as full sample to save time

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape, X_test.shape

plt.imshow(X_train[4], cmap=plt.cm.gray_r)

y_train[4]

X_trainval, X_test, y_trainval, y_test = train_test_split(X_test, y_test, stratify=y_test, test_size=0.4, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, stratify=y_trainval, test_size=0.5, random_state=369)


np.min(X_train), np.max(X_train)

X_train.dtype

X_trainval = X_trainval.astype('float32')
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

X_trainval /= 255
X_train /= 255
X_val /= 255
X_test /= 255

X_trainval.shape, X_train.shape, X_val.shape, X_test.shape

X_trainval = X_trainval.reshape((6000, 28, 28, 1))
X_train = X_train.reshape((3000, 28, 28, 1))
X_val = X_val.reshape((3000, 28, 28, 1))
X_test = X_test.reshape((4000, 28, 28, 1))


# convert class vectors to binary class matrices
y_trainval = to_categorical(y_trainval)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

y_test_original = y_test
y_test = to_categorical(y_test)

y_train.shape, y_val.shape, y_test.shape

set_my_seed()
def build_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
              activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    return model
    
model = build_model()
model.summary()

hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=128, shuffle=False)

hist.history.keys()

val_loss = hist.history['val_loss']
index_min = np.argmin(val_loss)
index_min

plt.plot(hist.history['loss'], 'k', label='Training Loss')
plt.plot(val_loss, 'b', label='Validation Loss')
plt.axvline(index_min, linestyle='--', color='k')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

val_accuracy = hist.history['val_accuracy']
np.max(val_accuracy)

index_max = np.argmax(val_accuracy)
index_max

plt.plot(hist.history['accuracy'], 'k', label='Training Accuracy')
plt.plot(val_accuracy, 'b', label='Validation Accuracy')
plt.axvline(index_max, linestyle='--', color='k')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

set_my_seed()
model = build_model()
model.fit(X_trainval, y_trainval, epochs=index_max+1, batch_size=128, shuffle=False)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
test_loss, test_accuracy 

prob = model.predict(X_test)
prob[0]

pred = model.predict_classes(X_test)
pred[:5]

table = pd.crosstab(y_test_original, pred, rownames=['Actual'], colnames=['Predicted'])
table

sns.heatmap(table, cmap='Blues', annot=True, fmt='d')
plt.tight_layout()



