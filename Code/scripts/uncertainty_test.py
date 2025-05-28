# %%
import numpy as np
import tensorflow as tf

M = 10  # Number of ensemble members
K = 20  # Number of components in each ensemble member

mus = np.random.randn(M, K)
sigmas = np.random.randn(M, K)
logit_pis = np.random.randn(M, K)

pis = np.array(tf.nn.softmax(logit_pis, axis=1))

member_means = (mus * pis).sum(axis=1)
member_means = member_means.reshape(-1, 1)
combined_mean = mus.mean()

total_variance = ((pis / M) * (sigmas**2 + (mus - combined_mean) ** 2)).sum()
aleatoric_variance = ((pis / M) * (sigmas**2 + (mus - member_means) ** 2)).sum()
epistemic_variance = ((member_means - combined_mean) ** 2).mean()

print("Total Variance:", total_variance)
print("Aleatoric Variance:", aleatoric_variance)
print("Epistemic Variance:", epistemic_variance)
print(
    "Sum of Aleatoric and Epistemic Variance:", aleatoric_variance + epistemic_variance
)
