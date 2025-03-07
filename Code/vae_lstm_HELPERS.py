import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf

def plot_loss(history):
    """
    Plot training and validation loss over epochs.

    Args:
    - history (tf.keras.callbacks.History): Training history object from model.fit()
    """
    plt.figure(figsize=(10, 5))
    
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    
    plt.title("VAE Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()


def visualize_latent_space(encoder, X, method="PCA", batch_size=32):
    """
    Visualize latent space embeddings using PCA or t-SNE.

    Args:
    - encoder (tf.keras.Model): Trained encoder model.
    - X (numpy array): Input data to encode.
    - method (str): "PCA" or "TSNE" for dimensionality reduction.
    - batch_size (int): Batch size for prediction.
    """
    z_mean, _, _ = encoder.predict(X, batch_size=batch_size)

    if method == "PCA":
        reducer = PCA(n_components=2)
        title = "PCA Projection of Latent Space"
    elif method == "TSNE":
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        title = "t-SNE Projection of Latent Space"
    else:
        raise ValueError("Invalid method. Use 'PCA' or 'TSNE'.")

    z_reduced = reducer.fit_transform(z_mean)

    plt.figure(figsize=(10, 5))
    plt.scatter(z_reduced[:, 0], z_reduced[:, 1], alpha=0.5)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid()
    plt.show()


def plot_reconstruction(vae, X, num_samples=5, batch_size=32):
    """
    Plot original vs reconstructed samples.

    Args:
    - vae (tf.keras.Model): Trained VAE model.
    - X (numpy array): Input data.
    - num_samples (int): Number of samples to visualize.
    - batch_size (int): Batch size for prediction.
    """
    indices = np.random.choice(len(X), num_samples, replace=False)
    X_selected = X[indices]

    reconstructed = vae.predict(X_selected, batch_size=batch_size)

    plt.figure(figsize=(12, num_samples * 3))
    for i in range(num_samples):
        plt.subplot(num_samples, 1, i + 1)
        plt.plot(X_selected[i].flatten(), label="Original", color="blue")
        plt.plot(reconstructed[i].flatten(), label="Reconstructed", color="red", linestyle="dashed")
        plt.legend()
        plt.title(f"Sample {i+1}: Original vs Reconstructed")
    
    plt.tight_layout()
    plt.show()

def plot_reconstruction_distributions(vae, X):
    """
    Plot original vs reconstructed dsitrubtions.

    Args:
    - vae (tf.keras.Model): Trained VAE model.
    - X (numpy array): Input data.
    """
    reconstructed = vae.predict(X)

    num_features = X.shape[1]
    for i in range(num_features):
        plt.figure(figsize=(10, 5))
        sns.kdeplot(X[:, -1, i], label="Original", fill=True, color="blue", alpha=0.5)
        sns.kdeplot(reconstructed[:, -1, i], label="Reconstructed", fill=True, color="red", alpha=0.5)
        plt.title(f"Feature {i+1}: Original vs Reconstructed")
        plt.legend()
        plt.grid()
        plt.show()


def compare_real_vs_generated(encoder, decoder, X, num_samples=1000, batch_size=32):
    """
    Compare the distribution of real and generated data.

    Args:
    - encoder (tf.keras.Model): Trained encoder model.
    - decoder (tf.keras.Model): Trained decoder model.
    - X (numpy array): Input data.
    - num_samples (int): Number of samples for comparison.
    - batch_size (int): Batch size for prediction.
    """
    z_mean, _, _ = encoder.predict(X[:num_samples], batch_size=batch_size)

    # Generate new data from randomly sampled latent space
    z_random = np.random.normal(size=(num_samples, z_mean.shape[1]))
    generated_data = decoder.predict(z_random, batch_size=batch_size)

    # Flatten sequences for comparison
    real_flattened = X[:num_samples].flatten()
    generated_flattened = generated_data.flatten()

    # Plot distributions
    plt.figure(figsize=(10, 5))
    sns.kdeplot(real_flattened, label="Real Data", fill=True, color="blue", alpha=0.5)
    sns.kdeplot(generated_flattened, label="Generated Data", fill=True, color="red", alpha=0.5)
    plt.title("Distribution of Real vs Generated Data")
    plt.legend()
    plt.grid()
    plt.show()
