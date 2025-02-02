"""
Functions for estimating the true population proportion based on a sample statistic.
"""

import math
import scipy.stats as st
import numpy as np
import numpy.typing as npt
from typing import Union, Tuple

from ..core import margin_of_error


def sample_size(
	moe: float, prop: float=0.5, popl_size: int=None, alpha: float=0.05
	) -> int:
	"""
	Calculates the minimum sample size needed to estimate the proportion of observations within 
	a population that have a given characteristic, while meeting a given constraint on precision.

	Parameters
	----------
		moe : float in (0, 1)
			The desired margin of error, or the half-length of the desired confidence interval.
		prop : float in (0, 1), default 0.5
			Prior assumption around the proportion of the characteristic. Defaults to 0.5, which returns
			the most conservatively (large) sample size.
		popl_size : int, default None
			The size of the population. When `None`, assume an infinite population, and ignore the finite
			population correction (FPC).
		alpha : float in (0, 1), default 0.05
			The desired alpha level (1 - confidence level), defaulting to 0.05, i.e a 95% CL.

	Returns
	-------
		n : int
			The sample size needed to observe a population proportion as large as `prop`, at alpha
			level `alpha`, with a margin of error `moe`. Round decimal values up to the nearest integer
			using `math.ceil()`.

	Notes
	-----
	Validated against: https://www.statskingdom.com/50_ci_sample_size.html
	
	When `popl_size=None`, equivalent to `statsmodels.stats.proportion.samplesize_confint_proportion()`:
	https://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.samplesize_confint_proportion.html
	
	Bounds are clipped to be in the interval [0, 1].

	References
	----------
	.. [*] https://online.stat.psu.edu/stat506/lesson/2/2.3

	"""

	# Find critical value (z-score)
	cv = st.norm.ppf(1-alpha/2)

	# If N is provided, assume a finite population and use the finite population correction
	if popl_size:
		num = popl_size*prop*(1-prop)
		denom = (popl_size-1)*(moe**2)/(cv**2) + prop*(1-prop)
	# Otherwise, assume an infinite population
	else:
		num = cv**2*prop*(1-prop)
		denom = moe**2

	# Return sample size
	# Handle fractional sizes by returning the ceiling
	return math.ceil(num/denom)


def confint(
	success_count: int, sample_size: int, alpha: float=0.05, method: str='wilson'
	) -> Tuple[float]:
	"""
	Calculates the confidence interval for an estimate of a population proportion.
	Parameters
	----------
		success_count : int
			The number of observations in the sample having the given characteristic.
		sample_size : int
			The total number of observations in the sample.
		alpha : float in (0, 1), default 0.5
			The desired alpha level (1 - confidence level), defaulting to 0.05, i.e a 95% CL.
		method : str in {'normal', 'wilson'}, default 'wilson'
			The method to use in calculating the confidence interval. Supported methods:
			- 'normal' : normal approximation
			- 'wilson' : Wilson score interval (without continuity correction)
	
	Returns
	-------
		ci_lower, ci_upper : tuple of floats
			The lower and upper bounds of the confidence interval around `prop`.
	
	Notes
	-----
	Validated normal approximation against: https://www.statskingdom.com/proportion-confidence-interval-calculator.html
	
	Equivalent to `statsmodels.stats.proportion.proportion_confint()` for available methods.

	General recommendations for `method` selection:

	- For large sample sizes, the Normal approximation can be sufficient. However, it can perform poorly when the sample size is small or the proportion is close to 0 or 1. It is simple to calculate but is generally not recommended except for large sample sizes (typically when both np and n(1-p) are greater than 5, where n is the sample size and p is the observed proportion).
	- The Wilson score interval (without continuity correction) is often recommended for most situations due to its balance between accuracy and conservatism. It performs well across a wide range of sample sizes and proportions.
	- The Wilson score interval (with continuity correction) can be used for small sample sizes or when an extra conservative estimate is desired, though it may be overly conservative.
	- The Clopper-Pearson interval is recommended when you want to ensure the interval is extremely reliable, regardless of the width of the interval, especially for very small samples or when regulatory requirements demand the most conservative approach.

	TODO: 
 	- Incorporate additional esimation methods, e.g. Clopper-Pearson
	
	References
	----------
	.. [*] Normal approximation: https://online.stat.psu.edu/stat506/lesson/2/2.2
	.. [*] Wilson score interval: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
	.. [*] Wilson score interval: https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm
	"""
	# Find the sample proportion
	prop = success_count/sample_size

	# Find critical value (z-score)
	cv = st.norm.ppf(1-alpha/2)

	# Find variance in p
	var = prop*(1-prop)/sample_size
	
	if method == 'normal':
		center = prop

		# Find the standard error
		se = math.sqrt(var)

		# Find margin of error
		moe = margin_of_error(cv, se)
	
	elif method == 'wilson':
		fail_count = sample_size-success_count

		# Square the z-score
		cv_sq = cv**2

		# Calculate the center of the interval
		center = (success_count+cv_sq/2)/(sample_size+cv_sq)

		# Find margin of error
		# The first of these formulas (commented out) is more succinct; the second is less susceptible to rounding errors when success_count=0
		# moe = cv/(1+cv_sq/sample_size) * math.sqrt(var + cv_sq/(4*sample_size**2))
		moe = cv/(sample_size+cv_sq)*math.sqrt((success_count*fail_count)/sample_size + cv_sq/4)
	
	else:
		raise NotImplementedError(f"method {method} is not available")
	
	# Find lower and upper bounds of CI
	ci_lower, ci_upper = center-moe, center+moe
	
	# Clips bounds to be in the interval [0, 1]
	ci_lower, ci_upper = np.clip(ci_lower, 0, 1), np.clip(ci_upper, 0, 1)

	return ci_lower, ci_upper


def strat_proportion(
	success_counts: Union[npt.NDArray[np.int64], 'pd.Series[np.int64]'],
	sample_sizes: Union[npt.NDArray[np.int64], 'pd.Series[np.int64]'], 
	strat_sizes: Union[npt.NDArray[np.int64], 'pd.Series[np.int64]']
	) -> np.float64:
	"""
	Calculate the weighted proportion within a stratified sample.

	Parameters
	----------
		success_counts : array-like of ints
			The number of observations in each sample having the given characteristic.
		sample_sizes : array-like of ints
			The sample size of each stratum.
		strat_sizes : array-like of ints
			The number of observations in each stratum, whose grand total equals the size of the population.

	Returns
	-------
		p : float
			The weighted proportion across the entire stratified sample.

	Notes
	-----

	References
	----------
	.. [*] Lohr, S.: "Sampling: Design and Analysis", 2nd ed., ch. 3 (93-95)
	.. [*] https://stattrek.com/survey-research/stratified-sampling-analysis
	"""
	props = success_counts/sample_sizes
	
	# Find total population size
	popl_size = np.sum(strat_sizes)
	
	# Find sampling fraction (stratum weights)
	weights = strat_sizes/popl_size

	return np.sum(weights*props)


def strat_confint(
	success_counts: Union[npt.NDArray[np.int64], 'pd.Series[np.int64]'], 
	sample_sizes: Union[npt.NDArray[np.int64], 'pd.Series[np.int64]'], 
	strat_sizes: Union[npt.NDArray[np.int64], 'pd.Series[np.int64]'], 
	alpha: float=0.05
	) -> Tuple[float]:
	"""
	In a stratified sample setting, calculates the confidence interval for an estimate of a population proportion.

	Parameters
	----------
		success_counts : array-like of ints
			The number of observations in each sample having the given characteristic.
		sample_sizes : array-like of ints
			The sample size of each stratum.
		strat_sizes : array-like of ints
			The number of observations in each stratum, whose grand total equals the size of the population.
		alpha : float in (0, 1), default 0.05
			The desired alpha level (1 - confidence level), defaulting to 0.05, i.e a 95% CL.

	Returns
	-------
		ci_lower, ci_upper : tuple of floats
			The lower and upper bounds of the confidence interval around the (weighted) sample proportion.

	Notes
	-----
	Validated SE against Lohr, S.: "Sampling: Design and Analysis", 2nd ed, example 3.4 (p. 94)
	
	Bounds are clipped to be in the interval [0, 1].

	References
	----------
	.. [*] Lohr, S.: "Sampling: Design and Analysis", 2nd ed., ch. 3 (pp. 93-95)
	.. [*] https://stattrek.com/survey-research/stratified-sampling-analysis
	"""
	props = success_counts/sample_sizes

	# Find critical value (z-score)
	cv = st.norm.ppf(1-alpha/2)

	# Find total population size
	popl_size = np.sum(strat_sizes)

	# Find weighted sample proportion
	prop = strat_proportion(success_counts, sample_sizes, strat_sizes)
	
	# Find variance of each stratum proportion
	var = sample_sizes/(sample_sizes-1) * props * (1-props)

	# Find standard error
	se = np.sqrt(np.sum(strat_sizes**2 * (1/popl_size)**2 * (1-sample_sizes/strat_sizes) * var/sample_sizes))

	# Find margin of error
	moe = margin_of_error(cv, se)

	# Find lower and upper bounds of CI
	ci_lower, ci_upper = prop-moe, prop+moe
	
	# Clips bounds to be in the interval [0, 1]
	ci_lower, ci_upper = np.clip(ci_lower, 0, 1), np.clip(ci_upper, 0, 1)

	return ci_lower, ci_upper
