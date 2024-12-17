import numpy as np
from scipy.stats import norm


def expected_improvement(mean, var, best_value):
    """
    Calculate the expected improvement for the next architecture.
    """
    std = np.sqrt(var)
    z = (mean - best_value) / std
    return (mean - best_value) * norm.cdf(z) + std * norm.pdf(z)

# A dictionary mapping acquisition function names to function objects
acquisition_functions = {
    'expected_improvement': expected_improvement,
}
