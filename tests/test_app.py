import numpy as np
import pandas as pd


def test_rolling_mean():
    data = np.array([1, 3, 5, 9, 2, 7, 14, 3, 12, 9, 5])
    s = pd.Series(data)
    ma = s.rolling(window=3, min_periods=1).median()
    expected = np.array([1, 2, 3, 5, 5, 7, 7, 7, 12, 9, 9], dtype=np.float32)

    assert np.array_equal(ma, expected)


def test_rolling_std():
    data = np.array([1, 3, 5, 9, 2, 7, 14, 3, 12, 9, 5])
    s = pd.Series(data)
    # minimum number of observations in window required to have a value
    min_periods = 3
    std = s.rolling(window=3, min_periods=min_periods).std()

    # first elements should not be present because of period parameter
    assert std[:min_periods-1].isna().all()


