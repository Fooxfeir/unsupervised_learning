"""
KL Divergence Computation for Probability Distributions
=======================================================

This module implements KL divergence computation between different types
of probability distributions, with focus on Gaussian and Gaussian mixtures.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
import seaborn as sns

def gaussian_kl_divergence(mu1, sigma1, mu2, sigma2):
    """
    Analytical KL divergence between two multivariate Gaussians.
    
    KL(N(mu1, sigma1) || N(mu2, sigma2))
    
    Parameters:
    -----------
    mu1, mu2 : array-like, shape (d,)
        Mean vectors
    sigma1, sigma2 : array-like, shape (d, d)
        Covariance matrices
    
    Returns:
    --------
    kl_div : float
        KL divergence value
    """
    d = len(mu1)
    
    # Ensure inputs are numpy arrays
    mu1, mu2 = np.array(mu1), np.array(mu2)
    sigma1, sigma2 = np.array(sigma1), np.array(sigma2)
    
    # Compute inverse of sigma2
    sigma2_inv = np.linalg.inv(sigma2)
    
    # Difference in means
    mu_diff = mu2 - mu1
    
    # Terms in KL divergence formula
    trace_term = np.trace(sigma2_inv @ sigma1)
    quad_term = mu_diff.T @ sigma2_inv @ mu_diff
    log_det_term = np.log(np.linalg.det(sigma2) / np.linalg.det(sigma1))
    
    kl_div = 0.5 * (trace_term + quad_term - d + log_det_term)
    
    return kl_div

def monte_carlo_kl_divergence(p_samples, q_log_prob_func, n_samples=1000):
    """
    Monte Carlo approximation of KL(p||q) using samples from p.
    
    KL(p||q) ≈ (1/N) Σ [log p(x_i) - log q(x_i)] where x_i ~ p
    
    Parameters:
    -----------
    p_samples : array-like, shape (n_samples, d)
        Samples from distribution p
    q_log_prob_func : callable
        Function that computes log q(x) for input x
    n_samples : int
        Number of samples to use (if less than available)
    
    Returns:
    --------
    kl_div : float
        Estimated KL divergence
    """
    if len(p_samples) > n_samples:
        # Subsample if we have too many samples
        indices = np.random.choice(len(p_samples), n_samples, replace=False)
        samples = p_samples[indices]
    else:
        samples = p_samples
    
    # This is a simplified version - in practice, we'd need log p(x) too
    # Here we assume we're comparing fitted models where we can compute both
    log_q_values = q_log_prob_func(samples)
    
    # For demonstration, we'll return the negative average log q
    # In a real scenario, you'd subtract log p(x) values
    return -np.mean(log_q_values)

def gmm_kl_divergence_mc(gmm1, gmm2, n_samples=10000):
    """
    Monte Carlo approximation of KL divergence between two Gaussian mixtures.
    
    Parameters:
    -----------
    gmm1, gmm2 : GaussianMixture objects
        Fitted Gaussian mixture models
    n_samples : int
        Number of Monte Carlo samples
    
    Returns:
    --------
    kl_div : float
        Estimated KL(gmm1 || gmm2)
    """
    # Sample from gmm1
    samples, _ = gmm1.sample(n_samples)
    
    # Compute log probabilities under both models
    log_p = gmm1.score_samples(samples)
    log_q = gmm2.score_samples(samples)
    
    # KL divergence estimate
    kl_div = np.mean(log_p - log_q)
    
    return kl_div

def cross_entropy(p_samples, q_log_prob_func):
    """
    Compute cross-entropy H(p, q) = -E_p[log q(x)].
    
    Parameters:
    -----------
    p_samples : array-like
        Samples from distribution p
    q_log_prob_func : callable
        Function computing log q(x)
    
    Returns:
    --------
    cross_ent : float
        Cross-entropy value
    """
    log_q_values = q_log_prob_func(p_samples)
    return -np.mean(log_q_values)

def entropy_gaussian(sigma):
    """
    Analytical entropy of multivariate Gaussian.
    
    H(X) = 0.5 * log((2πe)^d * |Σ|)
    
    Parameters:
    -----------
    sigma : array-like, shape (d, d)
        Covariance matrix
    
    Returns:
    --------
    entropy : float
        Differential entropy
    """
    d = sigma.shape[0]
    det_sigma = np.linalg.det(sigma)
    entropy = 0.5 * (d * np.log(2 * np.pi * np.e) + np.log(det_sigma))
    return entropy

class TorchGaussianMixture(nn.Module):
    """
    PyTorch implementation of Gaussian Mixture for gradient computation.
    """
    
    def __init__(self, n_components, n_features):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features
        
        # Parameters (learnable)
        self.weights = nn.Parameter(torch.ones(n_components) / n_components)
        self.means = nn.Parameter(torch.randn(n_components, n_features))
        self.log_vars = nn.Parameter(torch.zeros(n_components, n_features))  # Diagonal covariance
    
    def log_prob(self, x):
        """Compute log probability of samples."""
        # x: (batch_size, n_features)
        # Expand dimensions for broadcasting
        x = x.unsqueeze(1)  # (batch_size, 1, n_features)
        means = self.means.unsqueeze(0)  # (1, n_components, n_features)
        vars = torch.exp(self.log_vars).unsqueeze(0)  # (1, n_components, n_features)
        
        # Compute log probabilities for each component
        log_probs = -0.5 * (self.n_features * torch.log(2 * torch.tensor(np.pi)) +
                           torch.sum(self.log_vars, dim=1) +
                           torch.sum((x - means)**2 / vars, dim=2))
        
        # Add log weights and use log-sum-exp
        log_probs = log_probs + torch.log(torch.softmax(self.weights, dim=0))
        return torch.logsumexp(log_probs, dim=1)

def demonstrate_gaussian_kl():
    """Demonstrate KL divergence between Gaussians."""
    print("KL Divergence between Gaussians")
    print("=" * 40)
    
    # Define two 2D Gaussians
    mu1 = np.array([0, 0])
    sigma1 = np.array([[1, 0.5], [0.5, 1]])
    
    mu2 = np.array([1, 1])
    sigma2 = np.array([[1.5, -0.3], [-0.3, 0.8]])
    
    # Compute KL divergences
    kl_12 = gaussian_kl_divergence(mu1, sigma1, mu2, sigma2)
    kl_21 = gaussian_kl_divergence(mu2, sigma2, mu1, sigma1)
    
    print(f"KL(N1 || N2) = {kl_12:.4f}")
    print(f"KL(N2 || N1) = {kl_21:.4f}")
    print(f"Asymmetry: |KL(N1||N2) - KL(N2||N1)| = {abs(kl_12 - kl_21):.4f}")
    
    # Visualize the distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Create grid
    x = np.linspace(-3, 4, 100)
    y = np.linspace(-3, 4, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    # Evaluate PDFs
    rv1 = multivariate_normal(mu1, sigma1)
    rv2 = multivariate_normal(mu2, sigma2)
    pdf1 = rv1.pdf(pos)
    pdf2 = rv2.pdf(pos)
    
    # Plot distributions
    axes[0].contour(X, Y, pdf1, levels=8, colors='blue', alpha=0.7)
    axes[0].scatter(mu1[0], mu1[1], color='blue', s=100, marker='x', linewidth=3)
    axes[0].set_title('Distribution 1')
    axes[0].set_xlabel('X₁')
    axes[0].set_ylabel('X₂')
    axes[0].axis('equal')
    
    axes[1].contour(X, Y, pdf2, levels=8, colors='red', alpha=0.7)
    axes[1].scatter(mu2[0], mu2[1], color='red', s=100, marker='x', linewidth=3)
    axes[1].set_title('Distribution 2')
    axes[1].set_xlabel('X₁')
    axes[1].set_ylabel('X₂')
    axes[1].axis('equal')
    
    # Overlay both
    axes[2].contour(X, Y, pdf1, levels=8, colors='blue', alpha=0.7, label='N1')
    axes[2].contour(X, Y, pdf2, levels=8, colors='red', alpha=0.7, label='N2')
    axes[2].scatter(mu1[0], mu1[1], color='blue', s=100, marker='x', linewidth=3)
    axes[2].scatter(mu2[0], mu2[1], color='red', s=100, marker='x', linewidth=3)
    axes[2].set_title(f'Both Distributions\nKL(N1||N2)={kl_12:.3f}')
    axes[2].set_xlabel('X₁')
    axes[2].set_ylabel('X₂')
    axes[2].axis('equal')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    return kl_12, kl_21

def demonstrate_gmm_kl():
    """Demonstrate KL divergence between Gaussian mixtures."""
    print("\nKL Divergence between Gaussian Mixtures")
    print("=" * 50)
    
    # Generate data from two different mixtures
    np.random.seed(42)
    
    # Mixture 1: Two well-separated components
    data1 = np.vstack([
        np.random.multivariate_normal([-2, 0], [[0.5, 0], [0, 0.5]], 200),
        np.random.multivariate_normal([2, 0], [[0.5, 0], [0, 0.5]], 200)
    ])
    
    # Mixture 2: Three components with different structure
    data2 = np.vstack([
        np.random.multivariate_normal([0, 2], [[1, 0.3], [0.3, 1]], 150),
        np.random.multivariate_normal([-1, -1], [[0.8, -0.2], [-0.2, 0.8]], 150),
        np.random.multivariate_normal([1, -1], [[0.6, 0.1], [0.1, 0.6]], 100)
    ])
    
    # Fit Gaussian mixtures
    gmm1 = GaussianMixture(n_components=2, random_state=42)
    gmm2 = GaussianMixture(n_components=3, random_state=42)
    
    gmm1.fit(data1)
    gmm2.fit(data2)
    
    # Compute KL divergences using Monte Carlo
    kl_12 = gmm_kl_divergence_mc(gmm1, gmm2, n_samples=5000)
    kl_21 = gmm_kl_divergence_mc(gmm2, gmm1, n_samples=5000)
    
    print(f"KL(GMM1 || GMM2) ≈ {kl_12:.4f}")
    print(f"KL(GMM2 || GMM1) ≈ {kl_21:.4f}")
    print(f"Asymmetry: |KL(GMM1||GMM2) - KL(GMM2||GMM1)| = {abs(kl_12 - kl_21):.4f}")
    
    # Compute cross-entropies
    samples1, _ = gmm1.sample(1000)
    samples2, _ = gmm2.sample(1000)
    
    ce_12 = cross_entropy(samples1, gmm2.score_samples)
    ce_21 = cross_entropy(samples2, gmm1.score_samples)
    
    print(f"Cross-entropy H(GMM1, GMM2) ≈ {ce_12:.4f}")
    print(f"Cross-entropy H(GMM2, GMM1) ≈ {ce_21:.4f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot data and fitted models
    x_min = min(data1[:, 0].min(), data2[:, 0].min()) - 1
    x_max = max(data1[:, 0].max(), data2[:, 0].max()) + 1
    y_min = min(data1[:, 1].min(), data2[:, 1].min()) - 1
    y_max = max(data1[:, 1].max(), data2[:, 1].max()) + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 80),
                         np.linspace(y_min, y_max, 80))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # GMM1
    density1 = np.exp(gmm1.score_samples(grid_points)).reshape(xx.shape)
    axes[0].contour(xx, yy, density1, levels=8, colors='blue', alpha=0.7)
    axes[0].scatter(data1[:, 0], data1[:, 1], alpha=0.5, s=10, color='blue')
    axes[0].set_title('GMM1 (2 components)')
    axes[0].set_xlabel('X₁')
    axes[0].set_ylabel('X₂')
    
    # GMM2
    density2 = np.exp(gmm2.score_samples(grid_points)).reshape(xx.shape)
    axes[1].contour(xx, yy, density2, levels=8, colors='red', alpha=0.7)
    axes[1].scatter(data2[:, 0], data2[:, 1], alpha=0.5, s=10, color='red')
    axes[1].set_title('GMM2 (3 components)')
    axes[1].set_xlabel('X₁')
    axes[1].set_ylabel('X₂')
    
    # Overlay both
    axes[2].contour(xx, yy, density1, levels=6, colors='blue', alpha=0.7, label='GMM1')
    axes[2].contour(xx, yy, density2, levels=6, colors='red', alpha=0.7, label='GMM2')
    axes[2].set_title(f'Comparison\nKL(GMM1||GMM2)≈{kl_12:.3f}')
    axes[2].set_xlabel('X₁')
    axes[2].set_ylabel('X₂')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    return gmm1, gmm2, kl_12, kl_21

def demonstrate_gradient_computation():
    """Demonstrate gradient computation using PyTorch."""
    print("\nGradient Computation with PyTorch")
    print("=" * 40)
    
    # Create synthetic target data
    torch.manual_seed(42)
    target_data = torch.randn(500, 2) * 0.8 + torch.tensor([1.0, -0.5])
    
    # Initialize model
    model = TorchGaussianMixture(n_components=1, n_features=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    losses = []
    for epoch in range(200):
        optimizer.zero_grad()
        
        # Negative log-likelihood loss
        log_probs = model.log_prob(target_data)
        loss = -torch.mean(log_probs)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Plot training curve
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log-Likelihood')
    plt.title('Training Curve')
    plt.grid(True)
    
    # Visualize fitted model
    plt.subplot(1, 2, 2)
    plt.scatter(target_data[:, 0], target_data[:, 1], alpha=0.6, label='Data')
    
    # Plot learned components
    with torch.no_grad():
        weights = torch.softmax(model.weights, dim=0)
        means = model.means
        for i in range(model.n_components):
            plt.scatter(means[i, 0], means[i, 1], 
                       s=weights[i]*500, marker='x', linewidth=3,
                       label=f'Component {i+1}')
    
    plt.xlabel('X₁')
    plt.ylabel('X₂')
    plt.title('Fitted Model')
    plt.legend()
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Final model weights: {torch.softmax(model.weights, dim=0).detach().numpy()}")
    print(f"Final model means:\n{model.means.detach().numpy()}")

def kl_parameter_sensitivity():
    """Analyze how KL divergence changes with parameters."""
    print("\nKL Divergence Parameter Sensitivity")
    print("=" * 45)
    
    # Fix one Gaussian, vary parameters of the other
    mu1 = np.array([0, 0])
    sigma1 = np.array([[1, 0], [0, 1]])  # Unit Gaussian
    
    # Vary mean of second Gaussian
    mean_distances = np.linspace(0, 3, 20)
    kl_values_mean = []
    
    for d in mean_distances:
        mu2 = np.array([d, 0])
        sigma2 = sigma1.copy()
        kl = gaussian_kl_divergence(mu1, sigma1, mu2, sigma2)
        kl_values_mean.append(kl)
    
    # Vary scale of second Gaussian
    scales = np.linspace(0.5, 3, 20)
    kl_values_scale = []
    
    for s in scales:
        mu2 = mu1.copy()
        sigma2 = s**2 * sigma1
        kl = gaussian_kl_divergence(mu1, sigma1, mu2, sigma2)
        kl_values_scale.append(kl)
    
    # Plot sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(mean_distances, kl_values_mean, 'o-')
    axes[0].set_xlabel('Mean Distance')
    axes[0].set_ylabel('KL Divergence')
    axes[0].set_title('KL vs Mean Separation')
    axes[0].grid(True)
    
    axes[1].plot(scales, kl_values_scale, 'o-', color='red')
    axes[1].set_xlabel('Scale Factor')
    axes[1].set_ylabel('KL Divergence')
    axes[1].set_title('KL vs Covariance Scale')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"KL divergence grows as ~distance² for mean separation")
    print(f"KL divergence for scale s: includes log(s²) and s² terms")

def main():
    """Main function demonstrating KL divergence computations."""
    print("KL Divergence Computation Demonstration")
    print("=" * 50)
    
    # 1. Gaussian KL divergence
    print("1. Analytical KL between Gaussians...")
    kl_12, kl_21 = demonstrate_gaussian_kl()
    
    # 2. Gaussian mixture KL divergence
    print("\n2. Monte Carlo KL between Gaussian mixtures...")
    gmm1, gmm2, kl_12_mc, kl_21_mc = demonstrate_gmm_kl()
    
    # 3. Gradient computation
    print("\n3. Gradient computation with automatic differentiation...")
    demonstrate_gradient_computation()
    
    # 4. Parameter sensitivity
    print("\n4. Parameter sensitivity analysis...")
    kl_parameter_sensitivity()
    
    # Summary
    print(f"\nSummary:")
    print(f"- Analytical KL computation: exact for Gaussians")
    print(f"- Monte Carlo approximation: needed for complex distributions")
    print(f"- Automatic differentiation: enables gradient-based optimization")
    print(f"- KL divergence is asymmetric and sensitive to parameters")

if __name__ == "__main__":
    main()
