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

    m = svm_train(trainY, trainX, '-s 3 -t 3 -c 2')
    print('======')
    p_label, p_acc, p_val = svm_predict(testY, testX, m)
    print('======')
    sec = (np.sum((np.array(testY)*stdY - np.array(p_label)*stdY)**2) / (len(testY) - 1)) ** 0.5
    __tmp = (np.sum((np.array(testY) * stdY) ** 2) / (len(testY) - 1)) ** 0.5
    R2 = 1 - (sec / __tmp)**2
    print(R2)
    clf()
    plot(np.array(testY)*stdY+aveY, np.array(p_label)*stdY+aveY, 'o')
    plot([90, 98], [90, 98], '--')
    xlabel('True')
    ylabel("Predict")
    title('SVM model evaluation [c = 2]\n'+r'($SEC = ' + '%.4lf' % sec + '$, $R^2 = %.4lf$)' % R2)
    grid()
    # show()
    savefig('img/svm_result_t3_c2.png')
    print('======')
    print('True:')
    print(np.array(testY) * stdY + aveY)
    print('Predict:')
    print(np.array(p_label) * stdY + aveY)

    m = svm_train(trainY, trainX, '-s 3 -t 3 -c 1')
    print('======')
    p_label, p_acc, p_val = svm_predict(testY, testX, m)
    print('======')
    sec = (np.sum((np.array(testY)*stdY - np.array(p_label)*stdY)**2) / (len(testY) - 1)) ** 0.5
    __tmp = (np.sum((np.array(testY) * stdY) ** 2) / (len(testY) - 1)) ** 0.5
    R2 = 1 - (sec / __tmp)**2
    print(R2)
    clf()
    plot(np.array(testY)*stdY+aveY, np.array(p_label)*stdY+aveY, 'o')
    plot([90, 98], [90, 98], '--')
    xlabel('True')
    ylabel("Predict")
    title('SVM model evaluation [c = 1]\n'+r'($SEC = ' + '%.4lf' % sec + '$, $R^2 = %.4lf$)' % R2)
    grid()
    # show()
    savefig('img/svm_result_t3_c1.png')
    print('======')
    print('True:')
    print(np.array(testY) * stdY + aveY)
    print('Predict:')
    print(np.array(p_label) * stdY + aveY)

    m = svm_train(trainY, trainX, '-s 3 -t 3 -c 1.6')
    print('======')
    p_label, p_acc, p_val = svm_predict(testY, testX, m)
    print('======')
    sec = (np.sum((np.array(testY)*stdY - np.array(p_label)*stdY)**2) / (len(testY) - 1)) ** 0.5
    __tmp = (np.sum((np.array(testY) * stdY) ** 2) / (len(testY) - 1)) ** 0.5
    R2 = 1 - (sec / __tmp)**2
    print(R2)
    clf()
    plot(np.array(testY)*stdY+aveY, np.array(p_label)*stdY+aveY, 'o')
    plot([90, 98], [90, 98], '--')
    xlabel('True')
    ylabel("Predict")
    title('SVM model evaluation [c = 1.6]\n'+r'($SEC = ' + '%.4lf' % sec + '$, $R^2 = %.4lf$)' % R2)
    grid()
    # show()
    savefig('img/svm_result_t3_c1_6.png')
    print('======')
    print('True:')
    print(np.array(testY) * stdY + aveY)
    print('Predict:')
    print(np.array(p_label) * stdY + aveY)

    m = svm_train(trainY, trainX, '-s 3 -t 3 -c 1.2')
    print('======')
    p_label, p_acc, p_val = svm_predict(testY, testX, m)
    print('======')
    sec = (np.sum((np.array(testY)*stdY - np.array(p_label)*stdY)**2) / (len(testY) - 1)) ** 0.5
    __tmp = (np.sum((np.array(testY) * stdY) ** 2) / (len(testY) - 1)) ** 0.5
    R2 = 1 - (sec / __tmp)**2
    print(R2)
    clf()
    plot(np.array(testY)*stdY+aveY, np.array(p_label)*stdY+aveY, 'o')
    plot([90, 98], [90, 98], '--')
    xlabel('True')
    ylabel("Predict")
    title('SVM model evaluation [c = 1.2]\n'+r'($SEC = ' + '%.4lf' % sec + '$, $R^2 = %.4lf$)' % R2)
    grid()
    # show()
    savefig('img/svm_result_t3_c1_2.png')
    print('======')
    print('True:')
    print(np.array(testY) * stdY + aveY)
    print('Predict:')
    print(np.array(p_label) * stdY + aveY)

    m = svm_train(trainY, trainX, '-s 3 -t 3 -c 0.8')
    print('======')
    p_label, p_acc, p_val = svm_predict(testY, testX, m)
    print('======')
    sec = (np.sum((np.array(testY)*stdY - np.array(p_label)*stdY)**2) / (len(testY) - 1)) ** 0.5
    __tmp = (np.sum((np.array(testY) * stdY) ** 2) / (len(testY) - 1)) ** 0.5
    R2 = 1 - (sec / __tmp)**2
    print(R2)
    clf()
    plot(np.array(testY)*stdY+aveY, np.array(p_label)*stdY+aveY, 'o')
    plot([90, 98], [90, 98], '--')
    xlabel('True')
    ylabel("Predict")
    title('SVM model evaluation [c = 0.8]\n'+r'($SEC = ' + '%.4lf' % sec + '$, $R^2 = %.4lf$)' % R2)
    grid()
    # show()
    savefig('img/svm_result_t3_c0_8.png')
    print('======')
    print('True:')
    print(np.array(testY) * stdY + aveY)
    print('Predict:')
    print(np.array(p_label) * stdY + aveY)

    m = svm_train(trainY, trainX, '-s 3 -t 0')
    print('======')
    p_label, p_acc, p_val = svm_predict(testY, testX, m)
    print('======')
    sec = (np.sum((np.array(testY)*stdY - np.array(p_label)*stdY)**2) / (len(testY) - 1)) ** 0.5
    __tmp = (np.sum((np.array(testY) * stdY) ** 2) / (len(testY) - 1)) ** 0.5
    R2 = 1 - (sec / __tmp)**2
    print(R2)
    clf()
    plot(np.array(testY)*stdY+aveY, np.array(p_label)*stdY+aveY, 'o')
    plot([90, 98], [90, 98], '--')
    xlabel('True')
    ylabel("Predict")
    title('SVM model evaluation [linear]\n'+r'($SEC = ' + '%.4lf' % sec + '$, $R^2 = %.4lf$)' % R2)
    grid()
    # show()
    savefig('img/svm_result_t0.png')
    print('======')
    print('True:')
    print(np.array(testY) * stdY + aveY)
    print('Predict:')
    print(np.array(p_label) * stdY + aveY)

    m = svm_train(trainY, trainX, '-s 3 -t 1')
    print('======')
    p_label, p_acc, p_val = svm_predict(testY, testX, m)
    print('======')
    sec = (np.sum((np.array(testY)*stdY - np.array(p_label)*stdY)**2) / (len(testY) - 1)) ** 0.5
    __tmp = (np.sum((np.array(testY) * stdY) ** 2) / (len(testY) - 1)) ** 0.5
    R2 = 1 - (sec / __tmp)**2
    print(R2)
    clf()
    plot(np.array(testY)*stdY+aveY, np.array(p_label)*stdY+aveY, 'o')
    plot([90, 98], [90, 98], '--')
    xlabel('True')
    ylabel("Predict")
    title('SVM model evaluation [polynomial]\n'+r'($SEC = ' + '%.4lf' % sec + '$, $R^2 = %.4lf$)' % R2)
    grid()
    # show()
    savefig('img/svm_result_t1.png')
    print('======')
    print('True:')
    print(np.array(testY) * stdY + aveY)
    print('Predict:')
    print(np.array(p_label) * stdY + aveY)

    m = svm_train(trainY, trainX, '-s 3 -t 2')
    print('======')
    p_label, p_acc, p_val = svm_predict(testY, testX, m)
    print('======')
    sec = (np.sum((np.array(testY)*stdY - np.array(p_label)*stdY)**2) / (len(testY) - 1)) ** 0.5
    __tmp = (np.sum((np.array(testY) * stdY) ** 2) / (len(testY) - 1)) ** 0.5
    R2 = 1 - (sec / __tmp)**2
    print(R2)
    clf()
    plot(np.array(testY)*stdY+aveY, np.array(p_label)*stdY+aveY, 'o')
    plot([90, 98], [90, 98], '--')
    xlabel('True')
    ylabel("Predict")
    title('SVM model evaluation [radial basis function]\n'+r'($SEC = ' + '%.4lf' % sec + '$, $R^2 = %.4lf$)' % R2)
    grid()
    # show()
    savefig('img/svm_result_t2.png')
    print('======')
    print('True:')
    print(np.array(testY) * stdY + aveY)
    print('Predict:')
    print(np.array(p_label) * stdY + aveY)

    m = svm_train(trainY, trainX, '-s 3 -t 3')
    print('======')
    p_label, p_acc, p_val = svm_predict(testY, testX, m)
    print('======')
    sec = (np.sum((np.array(testY)*stdY - np.array(p_label)*stdY)**2) / (len(testY) - 1)) ** 0.5
    __tmp = (np.sum((np.array(testY) * stdY) ** 2) / (len(testY) - 1)) ** 0.5
    R2 = 1 - (sec / __tmp)**2
    print(R2)
    clf()
    plot(np.array(testY)*stdY+aveY, np.array(p_label)*stdY+aveY, 'o')
    plot([90, 98], [90, 98], '--')
    xlabel('True')
    ylabel("Predict")
    title('SVM model evaluation [sigmoid]\n'+'($SEC = %.4lf' % sec + '$, $R^2 = %.4lf$)' % R2)
    grid()
    # show()
    savefig('img/svm_result_t3.png')
    print('======')
    print('True:')
    print(np.array(testY) * stdY + aveY)
    print('Predict:')
    print(np.array(p_label) * stdY + aveY)


