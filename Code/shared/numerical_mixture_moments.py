import numpy as np


def numerical_mixture_moments(weights, mus, sigmas, num_points=2000, range_factor=5):
    """
    Numerically estimates mean, variance, skewness, and kurtosis of a Gaussian mixture model.

    Parameters:
        weights (array-like): Mixture component weights (should sum to 1).
        mus (array-like): Means of the Gaussian components.
        sigmas (array-like): Standard deviations of the Gaussian components.
        num_points (int): Number of grid points for numerical integration.
        range_factor (float): Determines integration range as ±range_factor * mixture_std.

    Returns:
        dict: Dictionary with estimated mean, std, skewness, and excess kurtosis.
    """

    # Ensure inputs are NumPy arrays
    weights = np.asarray(weights)
    mus = np.asarray(mus)
    sigmas = np.asarray(sigmas)

    # Compute mixture mean
    mean_mixture = np.sum(weights * mus)

    # Compute mixture variance
    variance_mixture = np.sum(weights * (sigmas**2 + (mus - mean_mixture) ** 2))
    std_mixture = np.sqrt(variance_mixture)

    # Define numerical integration range
    x_min = mean_mixture - range_factor * std_mixture
    x_max = mean_mixture + range_factor * std_mixture
    x = np.linspace(x_min, x_max, num_points)

    # Evaluate mixture PDF over the range
    pdf_vals = np.zeros_like(x)
    for w, m, s in zip(weights, mus, sigmas):
        pdf_vals += (
            w * (1.0 / (s * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - m) / s) ** 2)
        )

    # Normalize PDF (should integrate to ~1)
    norm_factor = np.trapezoid(pdf_vals, x)
    pdf_vals /= norm_factor

    # Compute numerical moments
    mean_approx = np.trapezoid(x * pdf_vals, x)
    variance_approx = np.trapezoid((x - mean_approx) ** 2 * pdf_vals, x)
    std_approx = np.sqrt(variance_approx)

    m3_approx = np.trapezoid((x - mean_approx) ** 3 * pdf_vals, x)
    skewness_approx = m3_approx / (std_approx**3)

    m4_approx = np.trapezoid((x - mean_approx) ** 4 * pdf_vals, x)
    kurtosis_approx = m4_approx / (std_approx**4)

    # Convert to excess kurtosis
    excess_kurtosis_approx = kurtosis_approx - 3

    return {
        "mean": mean_approx,
        "std": std_approx,
        "skewness": skewness_approx,
        "excess_kurtosis": excess_kurtosis_approx,
    }
