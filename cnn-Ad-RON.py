import random

from matplotlib.pyplot import *
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.callbacks import EarlyStopping
import keras
import pickle
import numpy as np

file = open('pickle.dat', 'rb')
nm = pickle.load(file)
Afb = pickle.load(file)
Ad = pickle.load(file)
RON = pickle.load(file)
file.close()

X = []
aveX = []
stdX = []
for x in Ad:
    avex = np.average(x)
    stdx = np.std(x)
    # X.append(np.array([afb]).T)
    X.append((np.array([x]).T - avex) / stdx)
    aveX.append(avex)
    stdX.append(stdx)
X = np.array(X)
Y = np.array(RON)
aveY = np.average(Y)
stdY = np.std(Y)
Y = (Y - aveY) / stdY

trainX = []
trainY = []
testX = []
testY = []

for i in range(len(X)):
    if i % 3:
        trainX.append(X[i, :])
        trainY.append(Y[i])
    else:
        testX.append(X[i, :])
        testY.append(Y[i])

trainX = np.array(trainX)
trainY = np.array(trainY)
testX = np.array(testX)
testY = np.array(testY)

model = keras.models.Sequential()
model.add(Conv1D(8, kernel_size=9, input_shape=(497, 1), activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(Conv1D(16, kernel_size=7, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(Conv1D(24, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(Conv1D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
model.summary()
model.compile(loss='mse', optimizer='Adam')

early_stopping = EarlyStopping(monitor='loss', patience=100)
model.fit(trainX, trainY, verbose=0, epochs=8888, callbacks=[early_stopping])

loss = model.evaluate(testX, testY, verbose=0)
result = model.predict(testX).flatten()
sec = (np.sum((np.array(testY) * stdY - np.array(result) * stdY) ** 2) / (len(testY) - 1)) ** 0.5
__tmp = (np.sum((np.array(testY) * stdY) ** 2) / (len(testY) - 1)) ** 0.5
R2 = 1 - (sec / __tmp) ** 2

print('======')
print('True:')
print(testY)
print('Predict:')
print(result)
plot(testY * stdY + aveY, (result * stdY + aveY), 'o')
plot([90, 98], [90, 98], '--')
grid()
title('CNN model evaluation\n' + r'($SEC = ' + '%.4lf' % sec + '$, $R^2 = %.4lf$)' % R2)

ylabel('Predict')
xlabel('True')
# show()
savefig('img/cnn_result.png')
model.save('cnn-model.h5')
