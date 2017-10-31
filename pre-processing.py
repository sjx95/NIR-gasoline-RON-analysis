from matplotlib.pylab import *
from svmutil import *
import scipy.io
import numpy as np
import logging


def load(mat_data_path="NIRExerciseData_Gasoline.mat"):
    mat_data = scipy.io.loadmat(mat_data_path)
    _nm = mat_data["nm"][0]
    _A = mat_data["A"]
    _RON = mat_data["RON"]
    return _nm, _A, _RON


def nir_plot(x, y, _title):
    clf()
    title(_title)
    xlabel("nm")
    ylabel("A")
    grid()
    for a in y:
        plot(x, a, linewidth=1)
    logging.info('RAW data image -> "img/' + _title + '.png".')
    savefig("img/" + _title + ".png")
    return


def polyfit_filter(x, y, h):
    def calc(_x):
        return coeffs[0] * _x * _x + coeffs[1] * _x + coeffs[2]

    __degree = 2
    ret_y = []
    for i in range(len(x)):
        l = r = i
        while (l > 0) & (abs(x[l] - x[i]) < h):
            l -= 1
        if abs(x[l] - x[i]) > h:
            l += 1
        while (r < len(x) - 1) & (abs(x[r] - x[i]) < h):
            r += 1
        if abs(x[l] - x[i]) > h:
            r -= 1
        coeffs = np.polyfit(x[l:r], y[l:r], __degree)
        ret_y.append(calc(nm[i]))
    return x, ret_y


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    nm, A, RON = load()
    nir_plot(nm, A, "NIR Data (RAW)")
    Af = []
    for a in A:
        _nm, af = polyfit_filter(nm, a, 5)
        Af.append(af)
    nir_plot(nm, Af, "NIR Data (after polyfit filer)")


def test():
    y, x = svm_read_problem('heart_scale')
    x = []
    y = []
    for i in range(-100, 100):
        x.append({1: i})
        y.append(i)
    # x = [{1: 0}, {1: 1}]
    # y = [0, 1]
    xx = [{1: 0}, {1: 200}]
    m = svm_train(y, x, '-s 3 -t 0')
    p_label, p_acc, p_val = svm_predict([0, 200], xx, m)
    print(p_label)
    print(p_acc)
    print(p_val)
