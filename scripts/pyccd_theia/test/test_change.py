from ccd.change import *


def test_adjustpeek():
    # Base
    test_dates = np.array([0, 1])
    defpeek = 6
    ans = 96
    assert ans == adjustpeek(test_dates, defpeek)

    # Single Landsat sensor
    test_dates = np.arange(16 * 5, step=16)
    defpeek = 6
    ans = 6
    assert ans == adjustpeek(test_dates, defpeek)

    # Double Landsat sensor
    test_dates = np.arange(8 * 5, step=8)
    defpeek = 6
    ans = 12
    assert ans == adjustpeek(test_dates, defpeek)

    # Single observation
    test_dates = np.array([0])
    defpeek = 6
    ans = 6
    assert ans == adjustpeek(test_dates, defpeek)

    # Empty list (should not happen)
    test_dates = np.array([])
    defpeek = 6
    ans = 6
    assert ans == adjustpeek(test_dates, defpeek)
