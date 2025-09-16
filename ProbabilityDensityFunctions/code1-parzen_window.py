"""
Parzen Window (Kernel Density Estimation) Implementation
========================================================

This module implements Parzen window density estimation for 2D data.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import seaborn as sns

def generate_2d_data():
    """Generate synthetic 2D data with multiple clusters."""
    np.random.seed(42)
    
    # Generate three clusters
    cluster1 = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], 100)
    cluster2 = np.random.multivariate_normal([-1, -1], [[0.8, -0.3], [-0.3, 0.8]], 80)
    cluster3 = np.random.multivariate_normal([0, 3], [[0.5, 0], [0, 1.2]], 60)
    
    data = np.vstack([cluster1, cluster2, cluster3])
    labels = np.hstack([np.zeros(100), np.ones(80), np.full(60, 2)])
    
    return data, labels

def parzen_window_estimation_hypersphere(data, bandwidth='auto', kernel='gaussian'):
    """
    Estimate PDF using Parzen window method with explicit hypersphere volume calculation.
    
    Parameters:
    -----------
    data : array-like, shape (n_samples, n_features)
        Input data
    bandwidth : float or 'auto'
        Bandwidth parameter for kernel
    kernel : str
        Kernel type ('gaussian', 'tophat', 'epanechnikov', etc.)
    
    Returns:
    --------
    estimator : function
        Function that estimates density at query points
    bandwidth : float
        The bandwidth used
    """
    from scipy.special import gamma
    
    if bandwidth == 'auto':
        # Use cross-validation to select bandwidth
        bandwidths = np.logspace(-1, 1, 20)
        kde_cv = GridSearchCV(
            KernelDensity(kernel=kernel),
            {'bandwidth': bandwidths},
            cv=5
        )
        kde_cv.fit(data)
        bandwidth = kde_cv.best_params_['bandwidth']
        print(f"Optimal bandwidth (hypersphere-based): {bandwidth:.3f}")
    
    n_samples, n_features = data.shape
    
    def gaussian_kernel_hypersphere(u):
        """Gaussian kernel normalized for hypersphere volume."""
        # Standard Gaussian kernel
        return np.exp(-0.5 * np.sum(u**2, axis=1))
    
    def estimate_density(query_points):
        """
        Estimate density at query points using hypersphere-based Parzen window.
        
        Parameters:
        -----------
        query_points : array-like, shape (n_queries, n_features)
            Points where to estimate density
        
        Returns:
        --------
        densities : array, shape (n_queries,)
            Estimated densities
        """
        if query_points.ndim == 1:
            query_points = query_points.reshape(1, -1)
        
        n_queries = query_points.shape[0]
        densities = np.zeros(n_queries)
        
        # Hypersphere volume normalization factor
        d = n_features
        volume_unit_hypersphere = (np.pi**(d/2)) / gamma(d/2 + 1)
        normalization = n_samples * (bandwidth**d) * volume_unit_hypersphere
        
        for i, query_point in enumerate(query_points):
            # Compute distances to all data points
            differences = (data - query_point) / bandwidth
            
            # Apply Gaussian kernel
            kernel_values = gaussian_kernel_hypersphere(differences)
            
            # Sum and normalize
            densities[i] = np.sum(kernel_values) / normalization
        
        return densities
    
    return estimate_density, bandwidth

def parzen_window_estimation(data, bandwidth='auto', kernel='gaussian'):
    """
    Estimate PDF using Parzen window method (sklearn implementation for comparison).
    
    Parameters:
    -----------
    data : array-like, shape (n_samples, n_features)
        Input data
    bandwidth : float or 'auto'
        Bandwidth parameter for kernel
    kernel : str
        Kernel type ('gaussian', 'tophat', 'epanechnikov', etc.)
    
    Returns:
    --------
    kde : KernelDensity object
        Fitted KDE model
    """
    if bandwidth == 'auto':
        # Use cross-validation to select bandwidth
        bandwidths = np.logspace(-1, 1, 20)
        kde_cv = GridSearchCV(
            KernelDensity(kernel=kernel),
            {'bandwidth': bandwidths},
            cv=5
        )
        kde_cv.fit(data)
        bandwidth = kde_cv.best_params_['bandwidth']
        print(f"Optimal bandwidth (sklearn): {bandwidth:.3f}")
    
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    kde.fit(data)
    
    return kde

def evaluate_kde_on_grid(kde, data, n_points=100):
    """
    Evaluate KDE on a regular grid for visualization.
    
    Parameters:
    -----------
    kde : KernelDensity object
        Fitted KDE model
    data : array-like
        Original data (used to determine grid bounds)
    n_points : int
        Number of grid points per dimension
    
    Returns:
    --------
    xx, yy : arrays
        Grid coordinates
    density : array
        Density values on grid
    """
    x_min, x_max = data[:, 0].min() - 2, data[:, 0].max() + 2
    y_min, y_max = data[:, 1].min() - 2, data[:, 1].max() + 2
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, n_points),
        np.linspace(y_min, y_max, n_points)
    )
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    log_density = kde.score_samples(grid_points)
    density = np.exp(log_density).reshape(xx.shape)
    
    return xx, yy, density

def plot_parzen_results(data, labels, kde, xx, yy, density):
    """
    Plot basic Parzen window results (kept for compatibility).
    
    Parameters:
    -----------
    data : array-like
        Original data points
    labels : array-like
        Data labels for coloring
    kde : KernelDensity object
        Fitted KDE model
    xx, yy, density : arrays
        Grid and density for contour plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Original data with true clusters
    scatter = axes[0].scatter(data[:, 0], data[:, 1], c=labels, alpha=0.7, cmap='viridis')
    axes[0].set_title('Original Data with True Clusters')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    plt.colorbar(scatter, ax=axes[0])
    
    # Plot 2: KDE contours with data points
    contour = axes[1].contour(xx, yy, density, levels=10, alpha=0.8)
    axes[1].scatter(data[:, 0], data[:, 1], alpha=0.5, s=20, color='red')
    axes[1].set_title('Parzen Window (KDE) Estimation')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    plt.colorbar(contour, ax=axes[1])
    
    # Plot 3: 3D surface plot
    ax3d = fig.add_subplot(133, projection='3d')
    surface = ax3d.plot_surface(xx, yy, density, cmap='viridis', alpha=0.8)
    ax3d.set_title('PDF as 3D Surface')
    ax3d.set_xlabel('Feature 1')
    ax3d.set_ylabel('Feature 2')
    ax3d.set_zlabel('Density')
    plt.colorbar(surface, ax=ax3d, shrink=0.5)
    
    plt.tight_layout()
    plt.show()

def plot_parzen_results_comparison(data, labels, kde_sklearn, hypersphere_estimator, bandwidth_hyper, xx, yy):
    """
    Plot comparison between sklearn KDE and hypersphere-based Parzen window.
    
    Parameters:
    -----------
    data : array-like
        Original data points
    labels : array-like
        Data labels for coloring
    kde_sklearn : KernelDensity object
        Fitted sklearn KDE model
    hypersphere_estimator : function
        Hypersphere-based density estimator
    bandwidth_hyper : float
        Bandwidth used for hypersphere method
    xx, yy : arrays
        Grid coordinates
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Evaluate both methods on grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # sklearn KDE
    sklearn_density = np.exp(kde_sklearn.score_samples(grid_points)).reshape(xx.shape)
    
    # Hypersphere-based Parzen
    hypersphere_density = hypersphere_estimator(grid_points).reshape(xx.shape)
    
    # Row 1: Original data and methods
    # Plot 1: Original data with true clusters
    scatter = axes[0, 0].scatter(data[:, 0], data[:, 1], c=labels, alpha=0.7, cmap='viridis')
    axes[0, 0].set_title('Original Data with True Clusters')
    axes[0, 0].set_xlabel('Feature 1')
    axes[0, 0].set_ylabel('Feature 2')
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # Plot 2: sklearn KDE
    contour1 = axes[0, 1].contour(xx, yy, sklearn_density, levels=10, alpha=0.8)
    axes[0, 1].scatter(data[:, 0], data[:, 1], alpha=0.5, s=20, color='red')
    axes[0, 1].set_title(f'sklearn KDE (h={kde_sklearn.bandwidth:.3f})')
    axes[0, 1].set_xlabel('Feature 1')
    axes[0, 1].set_ylabel('Feature 2')
    plt.colorbar(contour1, ax=axes[0, 1])
    
    # Plot 3: Hypersphere-based Parzen
    contour2 = axes[0, 2].contour(xx, yy, hypersphere_density, levels=10, alpha=0.8)
    axes[0, 2].scatter(data[:, 0], data[:, 1], alpha=0.5, s=20, color='red')
    axes[0, 2].set_title(f'Hypersphere Parzen (h={bandwidth_hyper:.3f})')
    axes[0, 2].set_xlabel('Feature 1')
    axes[0, 2].set_ylabel('Feature 2')
    plt.colorbar(contour2, ax=axes[0, 2])
    
    # Row 2: 3D surfaces
    # 3D surface plot - sklearn
    ax3d1 = fig.add_subplot(234, projection='3d')
    surface1 = ax3d1.plot_surface(xx, yy, sklearn_density, cmap='viridis', alpha=0.8)
    ax3d1.set_title('sklearn KDE - 3D Surface')
    ax3d1.set_xlabel('Feature 1')
    ax3d1.set_ylabel('Feature 2')
    ax3d1.set_zlabel('Density')
    
    # 3D surface plot - hypersphere
    ax3d2 = fig.add_subplot(235, projection='3d')
    surface2 = ax3d2.plot_surface(xx, yy, hypersphere_density, cmap='plasma', alpha=0.8)
    ax3d2.set_title('Hypersphere Parzen - 3D Surface')
    ax3d2.set_xlabel('Feature 1')
    ax3d2.set_ylabel('Feature 2')
    ax3d2.set_zlabel('Density')
    
    # Difference plot
    difference = np.abs(sklearn_density - hypersphere_density)
    contour_diff = axes[1, 2].contour(xx, yy, difference, levels=10, cmap='Reds')
    axes[1, 2].set_title('Absolute Difference')
    axes[1, 2].set_xlabel('Feature 1')
    axes[1, 2].set_ylabel('Feature 2')
    plt.colorbar(contour_diff, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.show()
    
    return sklearn_density, hypersphere_density

def compare_bandwidths(data):
    """
    Compare different bandwidth choices.
    
    Parameters:
    -----------
    data : array-like
        Input data
    """
    bandwidths = [0.1, 0.5, 1.0, 2.0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, bw in enumerate(bandwidths):
        kde = KernelDensity(bandwidth=bw, kernel='gaussian')
        kde.fit(data)
        
        xx, yy, density = evaluate_kde_on_grid(kde, data)
        
        contour = axes[i].contour(xx, yy, density, levels=8)
        axes[i].scatter(data[:, 0], data[:, 1], alpha=0.5, s=20)
        axes[i].set_title(f'Bandwidth = {bw}')
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function demonstrating Parzen window estimation."""
    print("Parzen Window (KDE) Demonstration")
    print("=" * 40)
    
    # Generate data
    data, labels = generate_2d_data()
    print(f"Generated {len(data)} data points in 2D")
    
    # Fit KDE with automatic bandwidth selection
    kde = parzen_window_estimation(data, bandwidth='auto')
    
    # Evaluate on grid
    xx, yy, density = evaluate_kde_on_grid(kde, data)
    
    # Plot results
    plot_parzen_results(data, labels, kde, xx, yy, density)
    
    # Compare different bandwidths
    print("\nComparing different bandwidth choices...")
    compare_bandwidths(data)
    
    # Compute log-likelihood on data
    log_likelihood = kde.score_samples(data)
    mean_ll = np.mean(log_likelihood)
    print(f"\nMean log-likelihood: {mean_ll:.4f}")
    
    # Sample from KDE (approximate)
    print("\nGenerating samples from estimated PDF...")
    n_samples = 200
    sample_indices = np.random.choice(len(data), n_samples)
    kde_samples = data[sample_indices] + np.random.normal(0, kde.bandwidth, (n_samples, 2))
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5, label='Original data')
    plt.title('Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(kde_samples[:, 0], kde_samples[:, 1], alpha=0.7, color='orange', label='KDE samples')
    plt.title('Samples from KDE')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
