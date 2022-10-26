"""
Need to find a new home for this.
"""

def f_score(precision, recall, beta=1):
    """
    Calculate the F-score for the given `precision`, `recall`, and `beta` values.

    Parameters
    ----------
        precision (float): The precision estimate, which must be a value in the range [0, 1].

        recall (float): The recall estimate, which must be a value in the range [0, 1].

        beta (int): The beta value to be used. Defaults to 1, in which case the return value is equivalent to the F1 score.
    """
    return (1+beta**2) * (precision*recall) / (beta**2*precision+recall)
