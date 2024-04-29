import numpy as np
from sklearn.linear_model import Lasso

def Adaptive_LASSO(X_train, y_train, lasso_iterations = 25, alpha = 1e-5, verbose=False):
    # set lists
    coefficients_list = []
    iterations_list   = []
    
    # set constants
    n_lasso_iterations = lasso_iterations
    
    g = lambda w: np.sqrt(np.abs(w))
    gprime = lambda w: 1. / (2. * np.sqrt(np.abs(w)) + np.finfo(float).eps)

    n_samples, n_features = X_train.shape
    p_obj = lambda w: 1. / (2 * n_samples) * np.sum((y_train - np.dot(X_train, w)) ** 2) \
                      + alpha * np.sum(g(w))
    weights = np.ones(n_features)

    X_w = X_train / weights[np.newaxis, :]
    X_w  = np.nan_to_num(X_w)
    X_w  = np.round(X_w,decimals = 3)

    y_train    = np.nan_to_num(y_train)

    adaptive_lasso = Lasso(alpha=alpha, fit_intercept=False)

    adaptive_lasso.fit(X_w, y_train)

    min_obj = np.inf
    for k in range(n_lasso_iterations):
        X_w = X_train / weights[np.newaxis, :]
        adaptive_lasso = Lasso(alpha=alpha, fit_intercept=False)
        adaptive_lasso.fit(X_w, y_train)
        coef_ = adaptive_lasso.coef_ / weights
        print(coef_)
        weights = gprime(coef_)
        p_obj_val = p_obj(coef_)
        if p_obj_val < min_obj:
            min_obj = p_obj_val
        else:
            break
        if verbose:
            print('Iteration #',k+1,':   ', min_obj)  # should go down
        iterations_list.append(k)
        coefficients_list.append(p_obj(coef_))
        
    return adaptive_lasso
