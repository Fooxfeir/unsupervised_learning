"""
k-Nearest Neighbors Density Estimation
======================================

This module implements k-NN based density estimation for 2D data.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.special import gamma
import seaborn as sns

def knn_density_estimation(data, k=5, h=1.0, use_exponential=True):
    """
    Estimate density using k-nearest neighbors approach with exponential kernel.
    
    Parameters:
    -----------
    data : array-like, shape (n_samples, n_features)
        Input data
    k : int
        Number of nearest neighbors
    h : float
        Bandwidth parameter for exponential decay (only used if use_exponential=True)
    use_exponential : bool
        If True, use exponential weighting; if False, use traditional hard counting
    
    Returns:
    --------
    density_estimator : function
        Function that estimates density at query points
    """
    if use_exponential:
        # Use exponential kernel approach (inline implementation)
        n_samples, n_features = data.shape
        
        # Fit k-NN model
        nbrs = NearestNeighbors(n_neighbors=k)
        nbrs.fit(data)
        
        def estimate_density(query_points):
            """
            Estimate density at query points using exponential weighting.
            """
            distances, _ = nbrs.kneighbors(query_points)
            
            # Apply exponential decay to distances
            weights = np.exp(-distances**2 / (2 * h**2))
            
            # Sum weights and normalize
            raw_density = np.sum(weights, axis=1)
            normalization = k * (h * np.sqrt(2 * np.pi)) ** n_features
            densities = raw_density / normalization
            
            return densities
        
        return estimate_density
    else:
        # Traditional k-NN density estimation (hard counting)
        return traditional_knn_density_estimation(data, k=k)

def traditional_knn_density_estimation(data, k=5):
    """
    Traditional k-NN density estimation with hard counting.
    
    Parameters:
    -----------
    data : array-like, shape (n_samples, n_features)
        Input data
    k : int
        Number of nearest neighbors
    
    Returns:
    --------
    density_estimator : function
        Function that estimates density at query points
    """
    n_samples, n_features = data.shape
    
    # Fit k-NN model
    nbrs = NearestNeighbors(n_neighbors=k+1)  # +1 because query point is included
    nbrs.fit(data)
    
    def estimate_density(query_points):
        """
        Estimate density at query points using traditional hard counting.
        
        Parameters:
        -----------
        query_points : array-like, shape (n_queries, n_features)
            Points where to estimate density
        
        Returns:
        --------
        densities : array, shape (n_queries,)
            Estimated densities
        """
        # Find k+1 nearest neighbors (including the query point itself if it's in data)
        distances, indices = nbrs.kneighbors(query_points)
        
        # Use the k-th nearest neighbor distance (skip the first one which might be the point itself)
        kth_distances = distances[:, k]  # k-th neighbor (0-indexed, so this is actually k+1-th)
        
        # Volume of d-dimensional hypersphere
        d = n_features
        volume_unit_sphere = (np.pi ** (d/2)) / gamma(d/2 + 1)
        volumes = volume_unit_sphere * (kth_distances ** d)
        
        # Density estimation: k / (N * V_k)
        densities = k / (n_samples * volumes)
        
        return densities
    
    return estimate_density

def hypersphere_volume(radius, dimension):
    """
    Calculate volume of d-dimensional hypersphere.
    
    Parameters:
    -----------
    radius : float or array
        Radius of hypersphere
    dimension : int
        Dimension of space
    
    Returns:
    --------
    volume : float or array
        Volume of hypersphere
    """
    return (np.pi ** (dimension/2)) / gamma(dimension/2 + 1) * (radius ** dimension)

def compare_k_values(data, k_values=[3, 5, 10, 20]):
    """
    Compare density estimation with different k values.
    
    Parameters:
    -----------
    data : array-like
        Input data
    k_values : list
        List of k values to compare
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    # Create grid for evaluation
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    for i, k in enumerate(k_values):
        # Fit k-NN density estimator
        density_func = knn_density_estimation(data, k=k)
        
        # Evaluate on grid
        densities = density_func(grid_points)
        density_grid = densities.reshape(xx.shape)
        
        # Plot
        contour = axes[i].contour(xx, yy, density_grid, levels=8)
        axes[i].scatter(data[:, 0], data[:, 1], alpha=0.6, s=20, color='red')
        axes[i].set_title(f'k-NN Density (k={k})')
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()

def analyze_dimensionality_effect():
    """
    Demonstrate how hypersphere volume changes with dimension.
    """
    dimensions = np.arange(1, 21)
    radius = 1.0
    
    volumes = [hypersphere_volume(radius, d) for d in dimensions]
    
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(dimensions, volumes, 'o-')
    plt.xlabel('Dimension')
    plt.ylabel('Volume of Unit Hypersphere')
    plt.title('Hypersphere Volume vs Dimension')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(dimensions, volumes, 'o-')
    plt.xlabel('Dimension')
    plt.ylabel('Volume (log scale)')
    plt.title('Hypersphere Volume (Log Scale)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print some specific values
    print("Hypersphere volumes for different dimensions:")
    for d in [1, 2, 3, 5, 10, 20]:
        vol = hypersphere_volume(1.0, d)
        print(f"Dimension {d:2d}: {vol:.6f}")

def knn_vs_parzen_comparison(data):
    """
    Compare k-NN and Parzen window methods.
    
    Parameters:
    -----------
    data : array-like
        Input data
    """
    from sklearn.neighbors import KernelDensity
    
    # Prepare grid
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 60),
                         np.linspace(y_min, y_max, 60))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # k-NN estimation
    knn_estimator = knn_density_estimation(data, k=20)
    knn_densities = knn_estimator(grid_points).reshape(xx.shape)
    
    # Parzen window estimation
    kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
    kde.fit(data)
    parzen_densities = np.exp(kde.score_samples(grid_points)).reshape(xx.shape)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original data
    axes[0].scatter(data[:, 0], data[:, 1], alpha=0.7)
    axes[0].set_title('Original Data')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    
    # k-NN
    contour1 = axes[1].contour(xx, yy, knn_densities, levels=10)
    axes[1].scatter(data[:, 0], data[:, 1], alpha=0.5, s=15, color='red')
    axes[1].set_title('k-NN Density (k=20)')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    
    # Parzen window
    contour2 = axes[2].contour(xx, yy, parzen_densities, levels=10)
    axes[2].scatter(data[:, 0], data[:, 1], alpha=0.5, s=15, color='red')
    axes[2].set_title('Parzen Window (h=0.5)')
    axes[2].set_xlabel('Feature 1')
    axes[2].set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
    
    return knn_densities, parzen_densities

def main():
    """Main function demonstrating k-NN density estimation."""
    print("k-Nearest Neighbors Density Estimation")
    print("=" * 45)
    
    # Generate synthetic data
    np.random.seed(42)
    cluster1 = np.random.multivariate_normal([1, 1], [[0.5, 0.2], [0.2, 0.5]], 100)
    cluster2 = np.random.multivariate_normal([-1, 1], [[0.3, -0.1], [-0.1, 0.4]], 80)
    data = np.vstack([cluster1, cluster2])
    
    print(f"Generated {len(data)} data points")
    
    # Demonstrate hypersphere volume effect
    print("\nAnalyzing hypersphere volume in different dimensions...")
    analyze_dimensionality_effect()
    
    # Compare different k values
    print(f"\nComparing different k values...")
    compare_k_values(data)
    
    # Compare k-NN vs Parzen window
    print(f"\nComparing k-NN vs Parzen window methods...")
    knn_densities, parzen_densities = knn_vs_parzen_comparison(data)
    
    # Analyze density at data points
    knn_estimator = knn_density_estimation(data, k=5)
    data_densities_knn = knn_estimator(data)
    
    from sklearn.neighbors import KernelDensity
    kde = KernelDensity(bandwidth=0.5)
    kde.fit(data)
    data_densities_parzen = np.exp(kde.score_samples(data))
    
    print(f"\nDensity statistics:")
    print(f"k-NN - Mean: {np.mean(data_densities_knn):.4f}, Std: {np.std(data_densities_knn):.4f}")
    print(f"Parzen - Mean: {np.mean(data_densities_parzen):.4f}, Std: {np.std(data_densities_parzen):.4f}")
    
    # Plot density histograms
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(data_densities_knn, bins=30, alpha=0.7, label='k-NN')
    plt.xlabel('Density')
    plt.ylabel('Frequency')
    plt.title('Distribution of k-NN Densities')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(data_densities_parzen, bins=30, alpha=0.7, color='orange', label='Parzen')
    plt.xlabel('Density')
    plt.ylabel('Frequency')
    plt.title('Distribution of Parzen Densities')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
