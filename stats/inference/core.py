def margin_of_error(cv: float, sd: float) -> float:
	"""
	Calculate the margin of error, given a critical value and a standard deviation estimate.
	
	Parameters
	----------
		cv : float
			The critical value.
		sd : float
			The standard deviation of the population parameter; typically estimated using the standard error (SE).

	Returns
	-------
		d : float
			The margin of error.

	Notes
	-----

	References
	----------

	"""

	return cv * sd
