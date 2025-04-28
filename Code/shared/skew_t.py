# %%
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Test settings
    n_samples = 100000
    nu = 8
    lam = 0.5

    # Generate samples
    samples = rvs_skewt(n_samples, nu=nu, lam=lam, random_state=42)

    # Plot histogram
    plt.hist(samples, bins=100, density=True, alpha=0.6, color="g")
    plt.title(f"Skewed t-distribution samples\nnu={nu}, lambda={lam}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()

    # Simple sanity checks
    print(f"Sample mean: {np.mean(samples):.4f}")
    print(f"Sample std dev: {np.std(samples):.4f}")
    print(
        f"Sample skewness: {((samples - np.mean(samples))**3).mean() / np.std(samples)**3:.4f}"
    )
