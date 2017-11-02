from matplotlib.pylab import *
import scipy.io
import numpy as np
import pickle


def load(mat_data_path="NIRExerciseData_Gasoline.mat"):
    mat_data = scipy.io.loadmat(mat_data_path)
    _nm = mat_data["nm"][0]
    _A = mat_data["A"]
    _RON = list(mat_data["RON"].flatten())
    return _nm, _A, _RON


def nir_plot(x, y, _title, xl='nm', yl='A', _file=None):
    clf()
    title(_title)
    xlabel(xl)
    ylabel(yl)
    grid()
    for a in y:
        plot(x, a, linewidth=1)
    if _file is None:
        _file = "img/" + _title + ".png"
    savefig(_file)
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


def baseline_correction(x, y, tolerance=0.01):
    tolerance **= 2
    xx = x
    yy = y.copy()
    while True:
        f = np.poly1d(np.polyfit(xx, yy, 1))
        mse = 0
        for i in range(len(xx)):
            mse += (f(xx[i])-yy[i])**2
        mse /= len(xx)
        if mse <= tolerance:
            break
        xxx = []
        yyy = []
        for i in range(len(xx)):
            if yy[i] <= f(xx[i]):
                xxx.append(xx[i])
                yyy.append(yy[i])
            elif (yy[i] - f(xx[i]))**2 <= mse:
                xxx.append(xx[i])
                yyy.append(yy[i])
        xx = xxx
        yy = yyy
    ret_y = y.copy()
    for i in range(len(x)):
        ret_y[i] = y[i] - f(x[i])
    return x, ret_y


def polyfit_diff(x, y, h):
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
        f = np.poly1d(np.polyfit(x[l:r], y[l:r], __degree)).deriv()
        ret_y.append(f(nm[i]))
    return x, ret_y


if __name__ == '__main__':
    nm, A, RON = load()
    nir_plot(nm, A, "NIR Data (RAW)")
    Af = []
    for a in A:
        _nm, af = polyfit_filter(nm, a, 5)
        Af.append(af)
    nir_plot(nm, Af, "NIR Data (after polyfit filer)")
    Afb = []
    for af in Af:
        nm, afb = baseline_correction(nm, af, 0.01)
        Afb.append(afb)
    nir_plot(nm, Afb, "NIR Data (after baseline correction)")
    Ad = []
    for a in A:
        _nm, ad = polyfit_diff(nm, a, 10)
        Ad.append(ad)
    nir_plot(nm, Ad, "NIR Data (after polyfit differentiation)", yl='dA')

    X = np.array(Ad)
    Y = RON.copy()
    corr = np.corrcoef(Y, X.T)[0, 1:]
    clf()
    subplot(2, 1, 1)
    title('Correlation Analysis')
    ylabel('dA')
    for x in X:
        plot(nm, x)
    grid()
    subplot(2, 1, 2)
    ylabel(r'$\rho_{XY}$')
    xlabel('nm')
    plot(nm, corr)
    grid()
    # show()
    savefig('img/Correlation Analysis.png')

    output = open('pickle.dat', 'wb')
    pickle.dump(nm, output)
    pickle.dump(Afb, output)
    pickle.dump(Ad, output)
    pickle.dump(RON, output)
    pickle.dump(corr, output)
    output.close()
