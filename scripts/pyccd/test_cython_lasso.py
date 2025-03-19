import numpy as np
import time
from sklearn import linear_model
from ccd.models.cython_lasso import CythonLasso
from ccd.models.lasso import coefficient_matrix, fitted_model, predict

# Generate sample data
np.random.seed(42)
n_samples = 1000
n_features = 7
avg_days_yr = 365.2425

# Create dates (ordinal days)
start_date = 737000  # Some arbitrary start date
dates = np.arange(start_date, start_date + n_samples)

# Generate coefficients matrix
X = coefficient_matrix(dates, avg_days_yr, 8)

# True coefficients
true_coef = np.array([0.1, 0.5, -0.3, 0.2, -0.1, 0.05, -0.05])

# Generate target with noise
y = X.dot(true_coef) + np.random.normal(0, 0.1, n_samples)

# Test sklearn Lasso
print("Testing sklearn Lasso:")
start_time = time.time()
sklearn_lasso = linear_model.Lasso(alpha=0.1, max_iter=1000)
sklearn_lasso.fit(X, y)
sklearn_time = time.time() - start_time
print(f"Time: {sklearn_time:.4f} seconds")
print(f"Coefficients: {sklearn_lasso.coef_}")
print(f"Score: {sklearn_lasso.score(X, y):.4f}")

# Test Cython Lasso
print("\nTesting Cython Lasso:")
start_time = time.time()
cython_lasso = CythonLasso(alpha=0.1, max_iter=1000)
cython_lasso.fit(X, y)
cython_time = time.time() - start_time
print(f"Time: {cython_time:.4f} seconds")
print(f"Coefficients: {cython_lasso.coef_}")
print(f"RMSE: {cython_lasso.rmse:.4f}")

# Test PyCCD fitted_model with Cython
print("\nTesting PyCCD fitted_model with Cython:")
start_time = time.time()
model = fitted_model(dates, y, 1000, avg_days_yr, 8, 0.1)
pyccd_time = time.time() - start_time
print(f"Time: {pyccd_time:.4f} seconds")

# Compare predictions
print("\nComparing predictions:")
test_dates = dates[:10]  # Use first 10 dates for testing
sklearn_pred = sklearn_lasso.predict(coefficient_matrix(test_dates, avg_days_yr, 8))
cython_pred = cython_lasso.predict(coefficient_matrix(test_dates, avg_days_yr, 8))
pyccd_pred = predict(model, test_dates, avg_days_yr)

print(f"sklearn predictions: {sklearn_pred}")
print(f"Cython predictions: {cython_pred}")
print(f"PyCCD predictions: {pyccd_pred}")

# Compare performance
print(f"\nSpeedup: {sklearn_time / cython_time:.2f}x faster than sklearn")