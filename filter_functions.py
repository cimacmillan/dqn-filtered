import numpy as np
from scipy.stats import norm


def keep(rolling_mean, rolling_variance, rolling_n, error, filter_value):
    return True


def most_common_removal_filter(rolling_mean, rolling_variance, rolling_n, error, filter_value):
    deviation = np.sqrt(rolling_variance / rolling_n)
    how_many_deviations = abs((error - rolling_mean) / deviation)
    normal = ((norm.cdf(how_many_deviations) - 0.5) * 2.0)
    return normal > filter_value


def most_common_keep_filter(rolling_mean, rolling_variance, rolling_n, error, filter_value):
    deviation = np.sqrt(rolling_variance / rolling_n)
    how_many_deviations = abs((error - rolling_mean) / deviation)
    normal = ((norm.cdf(how_many_deviations) - 0.5) * 2.0)
    return normal < (1.0 - filter_value)


def smallest_remove(rolling_mean, rolling_variance, rolling_n, error, filter_value):
    deviation = np.sqrt(rolling_variance / rolling_n)
    how_many_deviations = (error - rolling_mean) / deviation
    normal = norm.cdf(how_many_deviations)
    return normal > filter_value
