import numpy as np


def sener_ranking_model(y_actual, lower_bounds, VaR_alpha):
    """
    Implements Sener's ranking model with both violation space and safe space penalization.

    Parameters
    ----------
    y_actual : np.array
        Actual returns.
    lower_bounds : np.array
        Predicted lower bounds (VaR levels).
    VaR_alpha : float
        The confidence level of the VaR model (e.g., 0.95 or 0.99).

    Returns
    -------
    total_interaction : float
        Penalization measure for the violation space (Φ).
    safe_space_penalty : float
        Penalization measure for the safe space (Ψ).
    """

    alpha_factor = VaR_alpha / (1 - VaR_alpha)

    # Step 1: Violation Space Penalization Φ(x, VaR)
    violation_mask = y_actual < lower_bounds
    eps = np.zeros_like(y_actual)
    eps[violation_mask] = lower_bounds[violation_mask] - y_actual[violation_mask]

    # Cluster formation
    clusters = []
    cluster_starts = []

    in_cluster = False
    current_cluster = []
    current_cluster_start = None

    for i, v in enumerate(violation_mask):
        if v:
            if not in_cluster:
                in_cluster = True
                current_cluster = []
                current_cluster_start = i
            current_cluster.append(eps[i])
        else:
            if in_cluster:
                in_cluster = False
                clusters.append(current_cluster)
                cluster_starts.append(current_cluster_start)

    if in_cluster and len(current_cluster) > 0:
        clusters.append(current_cluster)
        cluster_starts.append(current_cluster_start)

    # Compute C_i for each cluster
    clusters_C = [np.prod([1.0 + e for e in c]) - 1.0 for c in clusters]

    # Compute total interaction measure Φ(x, VaR)
    total_interaction = 0.0
    n_clusters = len(clusters)
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            prod_i = np.prod([1.0 + e for e in clusters[i]])
            prod_j = np.prod([1.0 + e for e in clusters[j]])
            k_ij = float(max(1, cluster_starts[j] - cluster_starts[i]))
            total_interaction += (1.0 / k_ij) * (prod_i * prod_j - 1.0)

    total_interaction *= alpha_factor

    # Step 2: Safe Space Penalization Ψ(x, VaR)
    safe_space_mask = (y_actual > lower_bounds) & (y_actual < 0)
    safe_space_penalty = np.sum(
        (y_actual[safe_space_mask] - lower_bounds[safe_space_mask])
    )

    return total_interaction, safe_space_penalty
