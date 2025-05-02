# %%
import numpy as np
from arch.univariate.distribution import SkewStudent


def skewt_nll(y_true, vol_pred, nu, skew, mu=None, reduce=True):
    """
    Per-obs loop to handle time-varying nu/skew in arch’s SkewStudent.

    y_true : (n,) observed returns (same units your model was fit in)
    vol_pred: (n,) one-step-ahead σ_t
    nu      : (n,) or scalar η parameters
    skew    : (n,) or scalar λ parameters
    mu      : (n,) or scalar location (default zero)
    reduce  : if True, return scalar sum NLL; else per-obs vector
    """
    y = np.asarray(y_true)
    σ = np.asarray(vol_pred)
    ηs = np.asarray(nu)
    λs = np.asarray(skew)
    if mu is None:
        rs = y
    else:
        rs = y - np.asarray(mu)

    dist = SkewStudent()
    n = len(y)
    ll = np.empty(n, dtype=float)

    for i in range(n):
        # each call gets scalar η and λ, and length-1 arrays for resids/sigma2
        params = [float(ηs[i]), float(λs[i])]
        resids = np.array([rs[i]])
        sigma2 = np.array([σ[i] ** 2])
        # individual=True returns a length-1 array
        ll_i = dist.loglikelihood(
            parameters=params, resids=resids, sigma2=sigma2, individual=True
        )
        ll[i] = ll_i[0]

    nll_vec = -ll
    return nll_vec.sum() if reduce else nll_vec


def rvs_skewt(n, nu, lam, rng=None):
    """
    Draw from Hansen's (1994) skewed Student-t.
    Accepts either an RNG or a seed.
    """
    if isinstance(rng, np.random.Generator):
        gen = rng
    else:
        gen = np.random.default_rng(rng)
    # Hansen constants
    c = (nu - 2) / nu
    omega = np.sqrt(1 + 3 * lam**2 - 4 * lam**2 / (1 + lam**2))
    alpha = 2 / (1 + lam**2)
    beta = -lam / np.sqrt(1 + lam**2)

    z = gen.standard_t(df=nu, size=n)
    return omega * (alpha * (beta + z) / np.sqrt(1 + beta**2 + (z + beta) ** 2 * c))


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
