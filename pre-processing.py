from matplotlib.pylab import *
from svmutil import *
import scipy.io
import numpy as np
import logging


def load(mat_data_path = "NIRExerciseData_Gasoline.mat"):
    mat_data = scipy.io.loadmat(mat_data_path)
    _nm = mat_data["nm"][0]
    _A = mat_data["A"]
    _RON = mat_data["RON"]
    return _nm, _A, _RON


def nir_plot(x, y, _title = ''):
    clf()
    title(_title)
    xlabel("nm")
    ylabel("A")
    grid()
    for a in y:
        plot(x, a, linewidth=1)
    logging.info('RAW data image -> "img/'+_title+'.png".')
    savefig("img/"+_title+".png")
    return


def polyfit_filter(x, y, radius):
    __degree = 2
    def calc(x):
        return coeffs[0]*x*x+coeffs[1]*x+coeffs[2]
    coeffs = np.polyfit(x[0:radius * 2], y[0:radius * 2], __degree)
    ret_y = []
    for i in range(radius):
        ret_y.append(calc(nm[i]))
    for i in range(radius, len(x)-radius+1):
        coeffs = np.polyfit(x[i - radius:i + radius], y[i - radius:i + radius], 2)
        ret_y.append(calc(nm[i]))
    coeffs = np.polyfit(x[len(x)- 2 * radius:len(x)], y[len(x) - 2 * radius:len(x)], __degree)
    for i in range(len(x)-radius+1, len(x)):
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