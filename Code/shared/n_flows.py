import numpy as np
from numba import njit

@njit(fastmath=True, parallel=True)
def compute_confidence_intervals(samples, confidence_levels):
    """
    Computes confidence intervals for a batch of samples.

    Args:
        samples (np.ndarray): Shape (batch_size, num_samples), predicted sample distributions.
        confidence_levels (list): Confidence levels to calculate percentiles for.

    Returns:
        np.ndarray: Confidence intervals of shape (batch_size, num_conf_levels, 2).
    """
    batch_size = samples.shape[0]
    num_conf_levels = len(confidence_levels)
    intervals = np.zeros((batch_size, num_conf_levels, 2))

    for i in range(num_conf_levels):
        cl = confidence_levels[i]
        lower_q = (1 - cl) / 2 * 100
        upper_q = 100 - lower_q

        for j in range(batch_size):  # Parallelized with Numba
            sorted_samples = np.sort(samples[j])  # Ensure sorting is within Numba's supported ops
            intervals[j, i, 0] = np.percentile(sorted_samples, lower_q)
            intervals[j, i, 1] = np.percentile(sorted_samples, upper_q)

    return intervals