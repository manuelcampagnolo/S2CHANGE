import sys
sys.path.insert(0, 'C:/Users/scaetano/Downloads/pyccd-1.x/pyccd-1.x')
from ccd.math_utils import *

def test_euclidean_norm():
    arr = np.arange(5)
    ans = 30.0 ** .5

    assert ans == euclidean_norm(arr)


def test_sum_of_squares():
    arr = np.arange(5)
    ans = 30

    assert ans == sum_of_squares(arr)

    arr2d = np.arange(10).reshape(2, -1)
    ans2d = [30, 255]

    assert np.array_equal(ans2d, sum_of_squares(arr2d, 1))


def test_calc_rmse():
    actual = np.arange(5)
    pred = np.arange(5) + 1

    ans_resids = np.ones_like(actual) * -1
    ans_rmse = 1.0

    rmse, resids = calc_rmse(actual, pred)

    assert rmse == ans_rmse
    assert np.array_equal(ans_resids, resids)


def test_calc_median():
    arr = np.arange(5)
    ans = 2

    assert ans == calc_median(arr)


def test_calc_residuals():
    actual = np.arange(5)
    pred = np.arange(5) + 1

    ans_resids = np.ones_like(actual) * -1

    assert np.array_equal(ans_resids, calc_residuals(actual, pred))


def test_kelvin_to_celsius():
    pass


def test_check_variogram():
    test1 = np.array([np.nan])
    test2 = np.arange(6)

    assert check_variogram(test1) is False
    assert check_variogram(test2) is True


def test_calc_variogram():
    # Base test
    test_obs = np.arange(4).reshape(2, -1)
    ans = np.array([1, 1], dtype=float)
    assert np.array_equal(ans, calculate_variogram(test_obs))

    # Test some different spectral spacing
    test_obs = np.array([[1, 3] * 2,
                         [4, 1] * 2])
    ans = np.array([2, 3], dtype=float)
    assert np.array_equal(ans, calculate_variogram(test_obs))

    # Test empty 2-d array, the detect function should prevent any empty 1-d arrays
    test_obs = np.array([[]])
    # ans = np.array([np.nan])
    assert all(np.isnan(calculate_variogram(test_obs)))

    # Test single observation
    test_obs = np.array([[1]])
    # ans = np.array([np.nan])
    assert all(np.isnan(calculate_variogram(test_obs)))


def test_adjusted_variogram():
    # Base test
    test_obs = np.stack([np.arange(35),
                         np.arange(35)])
    test_dates = np.arange(35)
    ans = np.array([31, 31], dtype=float)
    assert np.array_equal(ans, adjusted_variogram(test_dates, test_obs))

    # Test single Landsat sensor spacing
    test_obs = np.array([[1, 2],
                         [6, 5]])
    test_dates = np.arange(16 * 2, step=16)
    ans = np.array([1, 1], dtype=float)
    assert np.array_equal(ans, adjusted_variogram(test_dates, test_obs))

    test_obs = np.array([[1, 2, 3],
                         [6, 5, 4]])
    test_dates = np.arange(16 * 3, step=16)
    ans = np.array([2, 2], dtype=float)
    assert np.array_equal(ans, adjusted_variogram(test_dates, test_obs))

    # Test empty 2-d array, the detect function should prevent any empty 1-d arrays
    test_obs = np.array([[]])
    test_dates = np.array([])
    # ans = np.array([np.nan])
    assert all(np.isnan(adjusted_variogram(test_dates, test_obs)))

    # Test single observation
    test_obs = np.array([[1],
                         [6]])
    test_dates = np.arange(16, step=16)
    # ans = np.array([np.nan])
    assert all(np.isnan(adjusted_variogram(test_dates, test_obs)))
