# %%
import numpy as np
import numpy as np


def rvs_skewt(n, nu, lam, random_state=None):
    """
    Generate random variates from Hansen's (1994) skewed Student-t distribution.

    Parameters
    ----------
    n : int
        Number of samples
    nu : float
        Degrees of freedom (>2)
    lam : float
        Skewness parameter (-1 < lam < 1)
    random_state : int, RandomState instance or None

    Returns
    -------
    samples : ndarray
    """
    rng = np.random.default_rng(random_state)

    # Constants from Hansen (1994)
    c = (nu - 2) / nu
    omega = np.sqrt(1 + 3 * lam**2 - (4 * lam**2) / (1 + lam**2))
    alpha = 2 / (1 + lam**2)
    beta = -lam / np.sqrt(1 + lam**2)

    # Generate standard t samples
    z = rng.standard_t(df=nu, size=n)

    # Warp according to skewness
    samples = omega * (alpha * (beta + z) / np.sqrt(1 + beta**2 + (z + beta) ** 2 * c))

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

    for lam in [-0.9, -0.5, 0, 0.5, 0.9]:
        samples = rvs_skewt(n_samples, nu=nu, lam=lam, random_state=42)
        skewness = ((samples - np.mean(samples)) ** 3).mean() / np.std(samples) ** 3
        print(f"Lambda {lam:+.1f}: Sample skewness {skewness:.4f}")
