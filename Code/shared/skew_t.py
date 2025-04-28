import numpy as np


def rvs_skewt(n, nu, lam, random_state=None):
    """
    Draws random samples from a Hansen (1994) skewed Student-t distribution.

    Parameters:
    ----------
    n : int
        Number of samples.
    nu : float
        Degrees of freedom (> 2).
    lam : float
        Skewness parameter (-1 < lam < 1).
    random_state : int, RandomState instance or None
        Controls randomness (for reproducibility).

    Returns:
    -------
    samples : np.ndarray
        Random samples from skewed t distribution.
    """
    rng = np.random.default_rng(random_state)

    # Generate standard Student-t
    z = rng.standard_t(df=nu, size=n)

    # Apply skewness transformation
    delta = lam / np.sqrt(1 + lam**2)
    u = delta * z + np.sqrt(1 - delta**2) * rng.standard_normal(size=n)

    # Scale correction for skewed-t
    c = np.sqrt(nu / (nu - 2))
    samples = c * u

    return samples
