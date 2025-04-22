# %%
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

# %%
# Synthetic data
np.random.seed(0)
N = 50_000
X = np.linspace(0, 10, N)
true_w = np.array([1.0, 2.0])  # intercept = 1.0, slope = 2.0
true_sigma2 = 1
X_design = np.column_stack([np.ones(N), X])
y = X_design @ true_w + np.sqrt(true_sigma2) * np.random.randn(N)

# %%
# Plot
plt.scatter(X, y, s=0.1)
plt.xlabel("X")
plt.ylabel("y")
plt.show()

# %%
# Prior hyperparameters for Normal-Inverse-Gamma prior
w0 = np.zeros(X_design.shape[1])
V0 = np.eye(X_design.shape[1]) * 10.0  # loose prior on weights
a0 = 1.0  # shape for IG
b0 = 1.0  # scale for IG

# %%
# Posterior update
# V0, w0, a0, b0 are prior parameters
XTX = X_design.T @ X_design
XTy = X_design.T @ y

V0_inv = np.linalg.inv(V0)
Vn = np.linalg.inv(V0_inv + XTX)
wN = Vn @ (V0_inv @ w0 + XTy)

# Degrees of freedom update
aN = a0 + N / 2.0

# bN update
residuals = y - X_design @ wN
bN = b0 + 0.5 * ((wN - w0).T @ V0_inv @ (wN - w0) + residuals.T @ residuals)

# %%
# Posterior predictive distribution for new input x_*
x_star = np.array([[1.0, 2.5]])

# Posterior predictive mean:
predictive_mean = x_star @ wN

# Posterior predictive variance:
# The predictive distribution is a Student-t:
# Scale factor s^2 = bN/(aN)
# Predictive variance for the t-distribution:
# var(y_*) = s^2 * (1 + x_* Vn x_*^T)
epistemic_component = x_star @ Vn @ x_star.T
posterior_mean_sigma2 = bN / (aN - 1)  # posterior mean of sigma^2

# The aleatoric part can be approximated by the posterior mean of sigma^2
aleatoric_var = posterior_mean_sigma2

# The total predictive variance includes both scale-up from the t-distribution form:
total_predictive_var = (bN / aN) * (1 + epistemic_component)

# To separate them in a way analogous to the known noise case:
# Consider that the noise is not a known fixed value but rather a distribution.
# The posterior mean sigma^2 acts as the aleatoric estimate:
# epistemic_var ~ (bN / aN) * (x_* Vn x_*^T)
# aleatoric_var ~ (bN / aN)  (this is the expected noise under the posterior)

epistemic_var = (bN / aN) * epistemic_component
aleatoric_var = bN / aN

print("Estimated weights:", wN)
print("True weights:", true_w)

print("Posterior predictive mean:", predictive_mean[0])
print("Total predictive variance:", total_predictive_var[0])
print("Epistemic variance (approx):", epistemic_var[0])
print("Aleatoric variance (approx):", aleatoric_var)

true_y = x_star @ true_w
print("True y:", true_y[0])
print("True sigma^2:", true_sigma2)

# %%
