from svmutil import *
import pickle


def convert_libsvm(x):
    _x = []
    for xx in x:
        _xx = {}
        for i in range(len(nm)):
            _xx[i + 1] = xx[i]
        _x.append(_xx)
    return _x


if __name__=='__main__':
    file = open('pickle.dat', 'rb')
    nm = pickle.load(file)
    Afb = pickle.load(file)
    Ad = pickle.load(file)
    RON = pickle.load(file)
    file.close()

    Ad = convert_libsvm(Ad)
    m = svm_train(RON, Ad, '-s 3 -t 0')
