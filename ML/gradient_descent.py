import math
import numpy as np

def package_ans(gd_vals):
    x, fs, xs = gd_vals
    return [x.tolist(), [fs[0], fs[-1]], [xs[0].tolist(), xs[-1].tolist()]]

def hinge_l_basic(data, labels, th, th0):
    """ data: can be collumn vector or d x n matrix
        labels: can be n x 1, 1 x 1, int, float
        th: params, d x 1
        th0: other param, 1x1 
    """
    v = labels * (np.dot(th.T, data) + th0)
    return np.where(v < 1, 1-v, 0) #max(0,1-v)

def gradient_descent(f, df, theta, step_size_fn, max_iter):
    """ f: objective function
        df: derivative of objective funtion
        theta: (d+1) x 1 column vector of parameters to optimize
        step_size_fn: funtion that takes max_iter to determine step size
        max_iter: maximum iterations before stopping algorithm 
    """
    thetas = []         #list of all thetas and theat0's 
    thetas.append(theta)
    fs = [f(theta)]      #list of all objective values
    eta = step_size_fn(max_iter)
    for _ in range(max_iter):
        theta_old = thetas[-1]
        theta_new = theta_old - eta * df(theta_old)
        thetas.append(theta_new)
        fs.append(f(theta_new))
    return theta_new, fs, thetas

def numerical_gradient(f, delta=0.001):
    """ f: objective function
        delta: interval used to approximate gradient of f
    """
    def df(x):
        d, _ = np.shape(x) 
        gradient = np.zeros((d, 1))
        for i in range(d):
            delta_vector = np.zeros((d, 1))
            delta_vector[i] = delta
            gradient[i] = (f(x + delta_vector) - f(x - delta_vector)) / (2*delta)
        return gradient
    return df


def svm_objective(data, labels, th, th0, lam):
    """ data is dxn, labels is 1xn, th is dx1, th0 is 1x1, lam is a scalar used in regularizor
        return: support vector objective with hinge loss
    """
    _, n = labels.shape
    return np.sum(hinge_l_basic(data, labels, th, th0))/n + lam * (np.sum(th**2))

# Returns the full gradient as a single vector (which includes both th, th0)
def svm_objective_gradient(data, labels, th, th0, lam):
    """ data is dxn, labels is 1xn, th is dx1, th0 is 1x1, lam is a scalar used in regularizor
        return: gradient of support vector objective with hinge loss
    """
    _, n = labels.shape
    v = (-1 * labels * data)

    d_svm_obj_th = np.sum(np.where(v > 0.0, v, 0.0), axis = 1, keepdims = True) / n + lam * 2 * th
    d_svm_obj_th0 = np.sum(np.where(labels < 0.0, -labels, 0.0), axis = 1, keepdims = True) / n 

    return np.vstack((d_svm_obj_th ,d_svm_obj_th0))

def svm_min_step_size_fn(i):
    """ i: maximum number of iterations
        determines step size in gradient decsent
    """
    return 2/(i+1)**0.5

def batch_svm_min(data, labels, lam):
    d, n = data.shape
    theta = np.zeros((d+1, 1))

    def f(theta):
        return svm_objective(data, labels, theta[:d], theta[-1:], lam)

    def df(theta):
        return svm_objective_gradient(data, labels, theta[:d], theta[-1:], lam)

    return gradient_descent(f, df, theta, svm_min_step_size_fn, 10)
