import math
import scipy.stats as st
import numpy as np


def sample_size(
	d: float, p: float=0.5, N: int=None, cl: float=0.95
	) -> int:
    """
    Calculates the minimum sample size needed to estimate the proportion of observations within 
    a population that have a given characteristic, while meeting a given constraint on precision.

    Parameters
    ----------
        d : float in (0, 1)
            The required margin of error, or the half-length of the desired confidence interval.
        p : float in (0, 1), default 0.5
            Prior assumption around the proportion of the characteristic. Defaults to 0.5, which returns
            the most conservatively (large) sample size.
        N : int, default None
            The size of the population. When `None`, assume an infinite population, and ignore the finite
            population correction (fpc).
        cl : float in (0, 1), default 0.5
            The desired confidence level, defaulting to 0.95 or 95%.

    Returns
    -------
        n : int
            The sample size needed to observe a population proportion as large as `p`, at confidence
            level `cl`, with a margine of error `d`. Round decimal values up to the nearest integer
            using `math.ceil()`.

    Notes
    -----
    When `N=None`, equivalent to `statsmodels.stats.proportion.samplesize_confint_proportion()`:
    https://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.samplesize_confint_proportion.html

    References
    ----------
    .. [*] https://online.stat.psu.edu/stat506/lesson/2/2.3

    """

    # Find alpha value
    alpha = 1-cl

    # Find z-score
    z = st.norm.ppf(1-alpha/2)

    # If N is provided, assume a finite population and use the finite population correction
    if N:
        num = N*p*(1-p)
        denom = (N-1)*(d**2)/(z**2) + p*(1-p)
    # Otherwise, assume an infinite population
    else:
        num = z**2*p*(1-p)
        denom = d**2

    # Return sample size
    # Handle fractional sizes by returning the ceiling
    return math.ceil(num/denom)


def confint(
	p: float, n: int, cl: float=0.95
	) -> tuple:
    """
    Calculates the confidence interval for an estimate of a population proportion, using
    the normal approximation.

    Parameters
    ----------
        p : float in (0, 1)
            The observed proportion of observations in a sample having a given characteristic.
        n : int
            The number of observations in the sample.
        cl : float in (0, 1), default 0.5
            The desired confidence level, defaulting to 0.95 or 95%.

    Returns
    -------
        ci_low, ci_upper : tuple
            The lower and upper bounds of the confidence interval around `p`.

    Notes
    -----
	  Equivalent to `statsmodels.stats.proportion.proportion_confint()` using `method='normal'`.

    References
    ----------
    .. [*] https://online.stat.psu.edu/stat506/lesson/2/2.2
    """

    # Find alpha value
    alpha = 1-cl

    # Find z-score
    z = st.norm.ppf(1-alpha/2)

    # Find variance in p
    var_p = p*(1-p)/n

    # Find margin of error
    d = z*math.sqrt(var_p)

    return p - d, p + d
