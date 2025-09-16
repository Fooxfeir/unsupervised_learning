"""
Multivariate Gaussian Analysis
==============================

This module demonstrates multivariate Gaussian distributions, covariance analysis,
and geometric interpretations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns
from matplotlib.patches import Ellipse

def create_gaussian_data(mean, cov, n_samples=1000):
    """
    Generate samples from multivariate Gaussian distribution.
    
    Parameters:
    -----------
    mean : array-like, shape (d,)
        Mean vector
    cov : array-like, shape (d, d)
        Covariance matrix
    n_samples : int
        Number of samples to generate
    
    Returns:
    --------
    samples : array, shape (n_samples, d)
        Generated samples
    """
    return np.random.multivariate_normal(mean, cov, n_samples)

def analyze_covariance_matrix(cov):
    """
    Analyze properties of a covariance matrix.
    
    Parameters:
    -----------
    cov : array-like, shape (d, d)
        Covariance matrix
    
    Returns:
    --------
    eigenvals : array
        Eigenvalues
    eigenvecs : array
        Eigenvectors
    analysis : dict
        Analysis results
    """
    # Eigendecomposition
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    
    # Sort by eigenvalue magnitude (descending)
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    analysis = {
        'determinant': np.linalg.det(cov),
        'trace': np.trace(cov),
        'condition_number': eigenvals[0] / eigenvals[-1] if eigenvals[-1] > 1e-10 else np.inf,
        'eigenvalues': eigenvals,
        'eigenvectors': eigenvecs,
        'total_variance': np.sum(eigenvals),
        'variance_ratios': eigenvals / np.sum(eigenvals)
    }
    
    return eigenvals, eigenvecs, analysis

def plot_gaussian_contours(mean, cov, ax=None, n_std=2, **kwargs):
    """
    Plot confidence ellipses for 2D Gaussian distribution.
    
    Parameters:
    -----------
    mean : array-like, shape (2,)
        Mean vector
    cov : array-like, shape (2, 2)
        Covariance matrix
    ax : matplotlib axis
        Axis to plot on
    n_std : int
        Number of standard deviations for ellipse
    """
    if ax is None:
        ax = plt.gca()
    
    # Eigendecomposition
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    
    # Calculate ellipse parameters
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    width = 2 * n_std * np.sqrt(eigenvals[0])
    height = 2 * n_std * np.sqrt(eigenvals[1])
    
    # Create ellipse
    ellipse = Ellipse(mean, width, height, angle=angle, **kwargs)
    ax.add_patch(ellipse)
    
    return ellipse

def demonstrate_covariance_effects():
    """
    Demonstrate how different covariance matrices affect the distribution shape.
    """
    # Different covariance matrices
    covariances = {
        'Identity': np.eye(2),
        'Scaled Identity': 2 * np.eye(2),
        'Diagonal': np.diag([3, 0.5]),
        'Positive Correlation': np.array([[1, 0.8], [0.8, 1]]),
        'Negative Correlation': np.array([[1, -0.8], [-0.8, 1]]),
        'Strong Anisotropy': np.array([[4, 0], [0, 0.25]])
    }
    
    mean = np.array([0, 0])
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, (name, cov) in enumerate(covariances.items()):
        # Generate samples
        samples = create_gaussian_data(mean, cov, 500)
        
        # Analyze covariance
        eigenvals, eigenvecs, analysis = analyze_covariance_matrix(cov)
        
        # Plot
        axes[i].scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=20)
        
        # Plot confidence ellipses
        for n_std in [1, 2, 3]:
            plot_gaussian_contours(mean, cov, axes[i], n_std=n_std, 
                                 fill=False, edgecolor='red', alpha=0.7, linewidth=2)
        
        # Plot principal axes
        for j, (val, vec) in enumerate(zip(eigenvals, eigenvecs.T)):
            axes[i].arrow(mean[0], mean[1], 
                         vec[0] * np.sqrt(val), vec[1] * np.sqrt(val),
                         head_width=0.1, head_length=0.1, fc=f'C{j}', ec=f'C{j}')
        
        axes[i].set_title(f'{name}\nλ₁={eigenvals[0]:.2f}, λ₂={eigenvals[1]:.2f}')
        axes[i].set_xlabel('X₁')
        axes[i].set_ylabel('X₂')
        axes[i].axis('equal')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def gaussian_pdf_evaluation():
    """
    Demonstrate PDF evaluation for multivariate Gaussian.
    """
    # Define Gaussian parameters
    mean = np.array([1, 2])
    cov = np.array([[2, 1], [1, 1.5]])
    
    # Create evaluation grid
    x = np.linspace(-3, 5, 100)
    y = np.linspace(-2, 6, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    # Evaluate PDF
    rv = multivariate_normal(mean, cov)
    pdf_values = rv.pdf(pos)
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 2D contour plot
    contour = axes[0].contour(X, Y, pdf_values, levels=15)
    axes[0].set_title('PDF Contours')
    axes[0].set_xlabel('X₁')
    axes[0].set_ylabel('X₂')
    axes[0].axis('equal')
    plt.colorbar(contour, ax=axes[0])
    
    # 3D surface plot
    ax3d = fig.add_subplot(132, projection='3d')
    surf = ax3d.plot_surface(X, Y, pdf_values, cmap='viridis', alpha=0.8)
    ax3d.set_title('PDF as 3D Surface')
    ax3d.set_xlabel('X₁')
    ax3d.set_ylabel('X₂')
    ax3d.set_zlabel('p(x)')
    plt.colorbar(surf, ax=ax3d, shrink=0.5)
    
    # Heatmap
    heatmap = axes[2].imshow(pdf_values, extent=[x.min(), x.max(), y.min(), y.max()], 
                           origin='lower', cmap='viridis', aspect='equal')
    axes[2].set_title('PDF Heatmap')
    axes[2].set_xlabel('X₁')
    axes[2].set_ylabel('X₂')
    plt.colorbar(heatmap, ax=axes[2])
    
    plt.tight_layout()
    plt.show()
    
    return X, Y, pdf_values

def diagonal_vs_full_covariance():
    """
    Compare diagonal vs full covariance matrix models.
    """
    # True data with correlation
    true_mean = np.array([0, 0])
    true_cov = np.array([[1, 0.7], [0.7, 1]])
    
    # Generate data
    data = create_gaussian_data(true_mean, true_cov, 300)
    
    # Fit models
    sample_mean = np.mean(data, axis=0)
    full_cov = np.cov(data.T)
    diag_cov = np.diag(np.diag(full_cov))  # Keep only diagonal elements
    
    # Compare log-likelihoods
    rv_full = multivariate_normal(sample_mean, full_cov)
    rv_diag = multivariate_normal(sample_mean, diag_cov)
    
    ll_full = np.sum(rv_full.logpdf(data))
    ll_diag = np.sum(rv_diag.logpdf(data))
    
    print("Diagonal vs Full Covariance Comparison")
    print("=" * 45)
    print(f"True covariance:\n{true_cov}")
    print(f"\nEstimated full covariance:\n{full_cov}")
    print(f"\nDiagonal approximation:\n{diag_cov}")
    print(f"\nLog-likelihood (full): {ll_full:.2f}")
    print(f"Log-likelihood (diagonal): {ll_diag:.2f}")
    print(f"Difference: {ll_full - ll_diag:.2f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original data
    axes[0].scatter(data[:, 0], data[:, 1], alpha=0.6)
    plot_gaussian_contours(sample_mean, true_cov, axes[0], fill=False, 
                          edgecolor='red', linewidth=2, label='True')
    axes[0].set_title('Data with True Distribution')
    axes[0].set_xlabel('X₁')
    axes[0].set_ylabel('X₂')
    axes[0].legend()
    axes[0].axis('equal')
    
    # Full covariance fit
    axes[1].scatter(data[:, 0], data[:, 1], alpha=0.6)
    plot_gaussian_contours(sample_mean, full_cov, axes[1], fill=False, 
                          edgecolor='green', linewidth=2, label='Full Cov')
    axes[1].set_title('Full Covariance Model')
    axes[1].set_xlabel('X₁')
    axes[1].set_ylabel('X₂')
    axes[1].legend()
    axes[1].axis('equal')
    
    # Diagonal covariance fit
    axes[2].scatter(data[:, 0], data[:, 1], alpha=0.6)
    plot_gaussian_contours(sample_mean, diag_cov, axes[2], fill=False, 
                          edgecolor='orange', linewidth=2, label='Diagonal Cov')
    axes[2].set_title('Diagonal Covariance Model')
    axes[2].set_xlabel('X₁')
    axes[2].set_ylabel('X₂')
    axes[2].legend()
    axes[2].axis('equal')
    
    plt.tight_layout()
    plt.show()

def mahalanobis_distance_demo():
    """
    Demonstrate Mahalanobis distance vs Euclidean distance.
    """
    # Create data with correlation
    mean = np.array([0, 0])
    cov = np.array([[2, 1.5], [1.5, 2]])
    data = create_gaussian_data(mean, cov, 200)
    
    # Test point
    test_point = np.array([2, 1])
    
    # Calculate distances from test point to all data points
    euclidean_distances = np.linalg.norm(data - test_point, axis=1)
    
    # Mahalanobis distances
    cov_inv = np.linalg.inv(cov)
    mahalanobis_distances = np.sqrt(np.sum((data - test_point) @ cov_inv * (data - test_point), axis=1))
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Euclidean distance
    scatter1 = axes[0].scatter(data[:, 0], data[:, 1], c=euclidean_distances, cmap='viridis')
    axes[0].scatter(test_point[0], test_point[1], color='red', s=100, marker='x', linewidth=3)
    axes[0].set_title('Euclidean Distance')
    axes[0].set_xlabel('X₁')
    axes[0].set_ylabel('X₂')
    plt.colorbar(scatter1, ax=axes[0])
    axes[0].axis('equal')
    
    # Mahalanobis distance
    scatter2 = axes[1].scatter(data[:, 0], data[:, 1], c=mahalanobis_distances, cmap='viridis')
    axes[1].scatter(test_point[0], test_point[1], color='red', s=100, marker='x', linewidth=3)
    plot_gaussian_contours(mean, cov, axes[1], n_std=1, fill=False, edgecolor='red', alpha=0.7)
    plot_gaussian_contours(mean, cov, axes[1], n_std=2, fill=False, edgecolor='red', alpha=0.7)
    axes[1].set_title('Mahalanobis Distance')
    axes[1].set_xlabel('X₁')
    axes[1].set_ylabel('X₂')
    plt.colorbar(scatter2, ax=axes[1])
    axes[1].axis('equal')
    
    plt.tight_layout()
    plt.show()

def parameter_estimation_demo():
    """
    Demonstrate maximum likelihood parameter estimation.
    """
    # True parameters
    true_mean = np.array([2, -1])
    true_cov = np.array([[1.5, 0.8], [0.8, 2]])
    
    # Generate increasing amounts of data
    sample_sizes = [10, 50, 100, 500, 1000]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    mean_errors = []
    cov_errors = []
    
    for i, n in enumerate(sample_sizes):
        if i >= len(axes):
            break
            
        # Generate data
        data = create_gaussian_data(true_mean, true_cov, n)
        
        # Estimate parameters
        est_mean = np.mean(data, axis=0)
        est_cov = np.cov(data.T)
        
        # Calculate errors
        mean_error = np.linalg.norm(est_mean - true_mean)
        cov_error = np.linalg.norm(est_cov - true_cov, 'fro')
        
        mean_errors.append(mean_error)
        cov_errors.append(cov_error)
        
        # Plot
        axes[i].scatter(data[:, 0], data[:, 1], alpha=0.6, s=20)
        plot_gaussian_contours(true_mean, true_cov, axes[i], fill=False, 
                              edgecolor='red', linewidth=2, label='True')
        plot_gaussian_contours(est_mean, est_cov, axes[i], fill=False, 
                              edgecolor='blue', linewidth=2, linestyle='--', label='Estimated')
        axes[i].set_title(f'N = {n}\nMean Error: {mean_error:.3f}')
        axes[i].set_xlabel('X₁')
        axes[i].set_ylabel('X₂')
        axes[i].legend()
        axes[i].axis('equal')
    
    # Plot convergence
    axes[5].loglog(sample_sizes, mean_errors, 'o-', label='Mean Error')
    axes[5].loglog(sample_sizes, cov_errors, 's-', label='Covariance Error')
    axes[5].set_xlabel('Sample Size')
    axes[5].set_ylabel('Estimation Error')
    axes[5].set_title('Parameter Estimation Convergence')
    axes[5].legend()
    axes[5].grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function demonstrating Gaussian analysis."""
    print("Multivariate Gaussian Distribution Analysis")
    print("=" * 50)
    
    # Demonstrate covariance effects
    print("1. Demonstrating covariance matrix effects...")
    demonstrate_covariance_effects()
    
    # PDF evaluation
    print("\n2. Gaussian PDF evaluation...")
    X, Y, pdf_values = gaussian_pdf_evaluation()
    
    # Diagonal vs full covariance
    print("\n3. Diagonal vs full covariance comparison...")
    diagonal_vs_full_covariance()
    
    # Mahalanobis distance
    print("\n4. Mahalanobis distance demonstration...")
    mahalanobis_distance_demo()
    
    # Parameter estimation convergence
    print("\n5. Parameter estimation convergence...")
    parameter_estimation_demo()
    
    # Example covariance analysis
    print("\n6. Detailed covariance analysis example...")
    example_cov = np.array([[3, 1.5], [1.5, 2]])
    eigenvals, eigenvecs, analysis = analyze_covariance_matrix(example_cov)
    
    print(f"Covariance matrix:\n{example_cov}")
    print(f"Eigenvalues: {eigenvals}")
    print(f"Eigenvectors:\n{eigenvecs}")
    print(f"Determinant: {analysis['determinant']:.3f}")
    print(f"Trace: {analysis['trace']:.3f}")
    print(f"Condition number: {analysis['condition_number']:.3f}")
    print(f"Variance explained by PC1: {analysis['variance_ratios'][0]:.1%}")
    print(f"Variance explained by PC2: {analysis['variance_ratios'][1]:.1%}")

if __name__ == "__main__":
    main()