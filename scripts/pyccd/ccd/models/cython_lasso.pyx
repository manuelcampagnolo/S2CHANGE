# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Cython implementation of Lasso regression for PyCCD.
This module provides a fast coordinate descent implementation optimized for time series.
"""
from libc.math cimport fabs
import numpy as np
cimport numpy as np

# Define floating type that can be either float or double
ctypedef fused floating:
    float
    double

cdef floating soft_threshold(floating z, floating alpha) nogil:
    """Soft thresholding operator for Lasso coordinate descent"""
    if z > alpha:
        return z - alpha
    elif z < -alpha:
        return z + alpha
    else:
        return 0.0

cdef floating diff_abs_sum(int n, floating[::1] w, floating[::1] w_old) nogil:
    """Calculate sum(abs(w - w_old)) for convergence checking"""
    cdef int i
    cdef floating s = 0.0
    for i in range(n):
        s += fabs(w[i] - w_old[i])
    return s

cdef floating column_dot_product(int n_samples, const floating[:, ::1] X, int j) nogil:
    """Calculate X[:, j].dot(X[:, j]) efficiently"""
    cdef int i
    cdef floating result = 0.0
    for i in range(n_samples):
        result += X[i, j] * X[i, j]
    return result

cdef void compute_residuals(int n_samples, int n_features, 
                           const floating[:, ::1] X, const floating[::1] y,
                           const floating[::1] w, floating[::1] residuals) nogil:
    """Compute residuals y - X.dot(w)"""
    cdef int i, j
    
    # Initialize residuals with y
    for i in range(n_samples):
        residuals[i] = y[i]
        
    # Subtract X.dot(w)
    for i in range(n_samples):
        for j in range(n_features):
            residuals[i] -= X[i, j] * w[j]

cdef floating compute_rmse(int n_samples, int num_params, const floating[::1] residuals) nogil:
    """Compute root mean squared error"""
    cdef int i
    cdef floating sum_sq = 0.0
    
    for i in range(n_samples):
        sum_sq += residuals[i] * residuals[i]
    
    return (sum_sq / (n_samples - num_params)) ** 0.5

def lasso_coordinate_descent(floating[:, ::1] X, floating[::1] y, 
                            floating alpha=1.0, int max_iter=1000, 
                            floating tol=1e-4, bint fit_intercept=True):
    """
    Lasso regression using coordinate descent.
    
    Parameters:
    -----------
    X : ndarray, shape (n_samples, n_features)
        Training data
    y : ndarray, shape (n_samples,)
        Target values
    alpha : float, default=1.0
        Constant that multiplies the L1 term
    max_iter : int, default=1000
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for the optimization
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model
        
    Returns:
    --------
    dict: Contains the model results:
        'coef_': ndarray of coefficients
        'intercept_': intercept term
        'n_iter': number of iterations
        'residuals': ndarray of residuals
        'rmse': root mean squared error
    """
    # Get dimensions
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef int i, j, iteration
    cdef floating diff
    cdef floating z_j, p_j, old_w_j
    cdef floating[::1] X_j_squared = np.zeros(n_features, dtype=np.float64)
    
    # Handle intercept
    cdef floating[::1] X_mean = np.zeros(n_features, dtype=np.float64)
    cdef floating y_mean = 0.0
    cdef floating intercept = 0.0
    
    # Initialize coefficient arrays
    cdef floating[::1] w = np.zeros(n_features, dtype=np.float64)
    cdef floating[::1] w_old = np.zeros(n_features, dtype=np.float64)
    cdef floating[::1] residuals = np.zeros(n_samples, dtype=np.float64)
    
    # Declare X_centered at function level
    cdef floating[:, ::1] X_centered = np.zeros((n_samples, n_features), dtype=np.float64)
    
    # Reference to data that will be used in coordinate descent
    cdef floating[:, ::1] X_work
    
    if fit_intercept:
        # Calculate means for centering
        y_mean = np.mean(y)
        for j in range(n_features):
            X_mean[j] = np.mean(X[:, j])
        
        # Center y for initial residuals
        for i in range(n_samples):
            residuals[i] = y[i] - y_mean
            
        # Create centered X for use in coordinate descent
        for i in range(n_samples):
            for j in range(n_features):
                X_centered[i, j] = X[i, j] - X_mean[j]
                
        # Use X_centered in coordinate descent
        X_work = X_centered
    else:
        # Just copy y to residuals initially
        for i in range(n_samples):
            residuals[i] = y[i]
            
        # Use original X in coordinate descent
        X_work = X
    
    # Precompute X_j^T * X_j for each feature using the working X
    for j in range(n_features):
        X_j_squared[j] = column_dot_product(n_samples, X_work, j)
    
    # Coordinate descent
    for iteration in range(max_iter):
        # Store previous coefficients for convergence check
        for j in range(n_features):
            w_old[j] = w[j]
        
        # Update each coordinate
        for j in range(n_features):
            if X_j_squared[j] == 0.0:
                continue
                
            # Current coefficient value
            old_w_j = w[j]
            
            # Compute partial residual
            p_j = 0.0
            for i in range(n_samples):
                p_j += X_work[i, j] * (residuals[i] + old_w_j * X_work[i, j])
            
            # Update coefficient using soft thresholding
            w[j] = soft_threshold(p_j, alpha) / X_j_squared[j]
            
            # Update residuals
            if w[j] != old_w_j:
                for i in range(n_samples):
                    residuals[i] -= (w[j] - old_w_j) * X_work[i, j]
        
        # Check for convergence
        diff = diff_abs_sum(n_features, w, w_old)
        if diff < tol:
            break
    
    # Calculate final RMSE
    cdef floating rmse = compute_rmse(n_samples, n_features, residuals)
    
    # Calculate intercept if needed
    if fit_intercept:
        intercept = y_mean
        for j in range(n_features):
            intercept -= w[j] * X_mean[j]
    
    # Convert memoryviews to numpy arrays for returning
    result = {
        'coef_': np.asarray(w),
        'intercept_': intercept,
        'n_iter': iteration + 1,
        'residuals': np.asarray(residuals),
        'rmse': rmse
    }
    
    return result

class CythonLasso:
    """
    Python wrapper class that behaves like sklearn's Lasso but uses
    the Cython implementation.
    """
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4, fit_intercept=True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0
        self.n_iter_ = None
        self.rmse = None
        self.residual = None
    
    def fit(self, X, y):
        """Fit model with coordinate descent"""
        # Ensure arrays are contiguous
        X_cont = np.ascontiguousarray(X, dtype=np.float64)
        y_cont = np.ascontiguousarray(y, dtype=np.float64)
        
        # Run the Cython coordinate descent
        result = lasso_coordinate_descent(
            X_cont, y_cont, 
            alpha=self.alpha, 
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept
        )
        
        self.coef_ = result['coef_']
        self.intercept_ = result['intercept_']
        self.n_iter_ = result['n_iter']
        self.rmse = result['rmse']
        self.residual = result['residuals']
        return self
    
    def predict(self, X):
        """Predict using the linear model"""
        if self.coef_ is None:
            raise ValueError("Model not fitted yet.")
        
        # Ensure array is contiguous
        X = np.ascontiguousarray(X, dtype=np.float64)
        return X.dot(self.coef_) + self.intercept_