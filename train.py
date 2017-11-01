import random
from matplotlib.pyplot import *
from svmutil import *
import pickle
import numpy as np


def convert_libsvm(x, selected):
    _x = []
    aveX = []
    stdX = []
    for i in range(len(x)):
        _x.append({})
    for i in range(len(x[0])):
        aveX.append(np.average(x[:, i]))
        stdX.append(np.std(x[:, i]))
        if selected[i]:
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
    corr = np.corrcoef(Y, X.T)[0, 1:]
    clf()
    subplot(2, 1, 1)
    title('Correlation Analysis')
    ylabel('dA')
    for x in X:
        plot(nm, x)
    subplot(2, 1, 2)
    ylabel(r'$\rho_{XY}$')
    xlabel('nm')
    plot(nm, corr)
    # show()
    savefig('img/Correlation Analysis.png')
    selected = []
    for i in range(len(nm)):
        if abs(corr[i]) > 0:
            selected.append(True)
        else:
            selected.append(False)

    X, aveX, stdX = convert_libsvm(X, selected)
    aveY = np.average(Y)
    stdY = np.std(Y)
    Y = (Y-aveY)/stdY

    trainX = []
    trainY = []
    testX = []
    testY = []
    for i in range(len(X)):
        if random.random() < 0.7:
            trainX.append(X[i])
            trainY.append(Y[i])
        else:
            testX.append(X[i])
            testY.append(Y[i])

    m = svm_train(trainY, trainX, '-s 3 -t 3')
    print('======')
    p_label, p_acc, p_val = svm_predict(testY, testX, m)
    print('======')
    print(p_acc)
    print(p_val)
    print(testY)
    print(p_label)
    clf()
    plot(np.array(testY)*stdY+aveY, '.')
    plot(np.array(p_label)*stdY+aveY, '*')
    show()
