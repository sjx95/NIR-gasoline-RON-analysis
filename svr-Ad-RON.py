import random
from matplotlib.pyplot import *
from svmutil import *
import pickle
import numpy as np
import os


def convert_libsvm(x):
    _x = []
    aveX = []
    stdX = []
    for i in range(len(x)):
        _x.append({})
    for i in range(len(x[0])):
        aveX.append(np.average(x[:, i]))
        stdX.append(np.std(x[:, i]))
        for j in range(len(x)):
            _x[j][i] = (x[j, i] - np.average(x[:, i])) / np.std(x[:, i])
    return _x, aveX, stdX


if __name__ == '__main__':
    file = open('pickle.dat', 'rb')
    nm = pickle.load(file)
    Afb = pickle.load(file)
    Ad = pickle.load(file)
    RON = pickle.load(file)
    file.close()

    X = np.array(Ad)
    Y = RON.copy()

    X, aveX, stdX = convert_libsvm(X)
    aveY = np.average(Y)
    stdY = np.std(Y)
    Y = (Y - aveY) / stdY

    trainX = []
    trainY = []
    testX = []
    testY = []
    for i in range(len(X)):
        if i % 3:
            trainX.append(X[i])
            trainY.append(Y[i])
        else:
            testX.append(X[i])
            testY.append(Y[i])

    m = svm_train(trainY, trainX, '-s 3 -t 3 -c 0.9')
    print('======')
    p_label, p_acc, p_val = svm_predict(testY, testX, m)
    print('======')
    mse = np.average((np.array(testY)*stdY - np.array(p_label)*stdY)**2)
    print('MSE = ', mse)
    clf()
    plot(np.array(testY)*stdY+aveY, '.', label='True')
    plot(np.array(p_label)*stdY+aveY, '*', label='Predict')
    legend()
    ylabel("RON")
    title('SVM model evaluation\n'+r'($MSE = ' + '%.4lf' % mse + r'$)')
    grid()
    # show()
    savefig('img/svm_result.png')
    print('======')
    print('True:')
    print(np.array(testY) * stdY + aveY)
    print('Predict:')
    print(np.array(p_label) * stdY + aveY)
