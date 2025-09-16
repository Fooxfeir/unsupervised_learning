"""
Curse of Dimensionality Demonstration
====================================

This module demonstrates various aspects of the curse of dimensionality
in high-dimensional spaces and its impact on density estimation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from sklearn.neighbors import NearestNeighbors
import seaborn as sns

def hypersphere_volume(dimension, radius=1.0):
    """
    Calculate volume of n-dimensional hypersphere.
    
    V_n(r) = π^(n/2) / Γ(n/2 + 1) * r^n
    
    Parameters:
    -----------
    dimension : int
        Dimension of space
    radius : float
        Radius of hypersphere
    
    Returns:
    --------
    volume : float
        Volume of hypersphere
    """
    return (np.pi ** (dimension / 2)) / gamma(dimension / 2 + 1) * (radius ** dimension)

def hypercube_volume(dimension, side_length=2.0):
    """
    Calculate volume of n-dimensional hypercube.
    
    Parameters:
    -----------
    dimension : int
        Dimension of space
    side_length : float
        Side length of hypercube
    
    Returns:
    --------
    volume : float
        Volume of hypercube
    """
    return side_length ** dimension

def analyze_volume_concentration():
    """
    Analyze how hypersphere volume concentrates as dimension increases.
    """
    print("Volume Concentration Analysis")
    print("=" * 35)
    
    dimensions = np.arange(1, 51)
    sphere_volumes = [hypersphere_volume(d) for d in dimensions]
    cube_volumes = [hypercube_volume(d) for d in dimensions]
    volume_ratios = np.array(sphere_volumes) / np.array(cube_volumes)
    
    # Find dimension where sphere volume peaks
    peak_dim = dimensions[np.argmax(sphere_volumes)]
    peak_volume = max(sphere_volumes)
    
    print(f"Peak sphere volume: {peak_volume:.3f} at dimension {peak_dim}")
    print(f"Volume at d=1: {sphere_volumes[0]:.3f}")
    print(f"Volume at d=10: {sphere_volumes[9]:.6f}")
    print(f"Volume at d=20: {sphere_volumes[19]:.10f}")
    print(f"Volume at d=50: {sphere_volumes[49]:.2e}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Linear scale
    axes[0, 0].plot(dimensions, sphere_volumes, 'o-', label='Hypersphere')
    axes[0, 0].axvline(x=peak_dim, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('Dimension')
    axes[0, 0].set_ylabel('Volume')
    axes[0, 0].set_title('Hypersphere Volume (Linear Scale)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Log scale
    axes[0, 1].semilogy(dimensions, sphere_volumes, 'o-', color='orange')
    axes[0, 1].axvline(x=peak_dim, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].set_xlabel('Dimension')
    axes[0, 1].set_ylabel('Volume (log scale)')
    axes[0, 1].set_title('Hypersphere Volume (Log Scale)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Volume ratio
    axes[1, 0].plot(dimensions, volume_ratios, 'o-', color='green')
    axes[1, 0].set_xlabel('Dimension')
    axes[1, 0].set_ylabel('Sphere/Cube Volume Ratio')
    axes[1, 0].set_title('Volume Ratio: Sphere/Cube')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Focus on first 15 dimensions
    axes[1, 1].plot(dimensions[:15], sphere_volumes[:15], 'o-', color='purple')
    axes[1, 1].axvline(x=peak_dim, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Dimension')
    axes[1, 1].set_ylabel('Volume')
    axes[1, 1].set_title('Hypersphere Volume (d ≤ 15)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return dimensions, sphere_volumes, volume_ratios

def distance_concentration_demo():
    """
    Demonstrate distance concentration in high dimensions.
    """
    print("\nDistance Concentration in High Dimensions")
    print("=" * 50)
    
    dimensions = [1, 2, 5, 10, 20, 50, 100]
    n_samples = 1000
    
    results = []
    
    for d in dimensions:
        # Generate samples from unit Gaussian
        samples = np.random.multivariate_normal(np.zeros(d), np.eye(d), n_samples)
        
        # Compute distances from origin
        distances = np.linalg.norm(samples, axis=1)
        
        # Statistics
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        relative_std = std_dist / mean_dist
        
        results.append({
            'dimension': d,
            'mean_distance': mean_dist,
            'std_distance': std_dist,
            'relative_std': relative_std,
            'theoretical_mean': np.sqrt(d),  # E[||x||] ≈ √d for large d
            'distances': distances
        })
        
        print(f"d={d:3d}: mean={mean_dist:.3f}, std={std_dist:.3f}, "
              f"rel_std={relative_std:.4f}, theory={np.sqrt(d):.3f}")
    
    # Plot distance distributions for selected dimensions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    dims_to_plot = [1, 2, 5, 10, 50, 100]
    for i, d in enumerate(dims_to_plot):
        if i < len(axes):
            result = next(r for r in results if r['dimension'] == d)
            distances = result['distances']
            
            axes[i].hist(distances, bins=50, alpha=0.7, density=True)
            axes[i].axvline(result['mean_distance'], color='red', linestyle='-', 
                           label=f'Mean: {result["mean_distance"]:.2f}')
            axes[i].axvline(result['theoretical_mean'], color='orange', linestyle='--',
                           label=f'Theory: {result["theoretical_mean"]:.2f}')
            axes[i].set_title(f'Dimension {d}\nRel. Std: {result["relative_std"]:.3f}')
            axes[i].set_xlabel('Distance from Origin')
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Plot convergence of relative standard deviation
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    dims = [r['dimension'] for r in results]
    rel_stds = [r['relative_std'] for r in results]
    theoretical_rel_std = [1/np.sqrt(2*d) for d in dims]  # σ/μ ≈ 1/√(2d)
    
    plt.plot(dims, rel_stds, 'o-', label='Observed')
    plt.plot(dims, theoretical_rel_std, '--', label='Theory: 1/√(2d)')
    plt.xlabel('Dimension')
    plt.ylabel('Relative Standard Deviation')
    plt.title('Distance Concentration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    mean_dists = [r['mean_distance'] for r in results]
    theoretical_means = [r['theoretical_mean'] for r in results]
    
    plt.plot(dims, mean_dists, 'o-', label='Observed')
    plt.plot(dims, theoretical_means, '--', label='Theory: √d')
    plt.xlabel('Dimension')
    plt.ylabel('Mean Distance from Origin')
    plt.title('Mean Distance Growth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results

def nearest_neighbor_analysis():
    """
    Analyze nearest neighbor distances in high dimensions.
    """
    print("\nNearest Neighbor Distance Analysis")
    print("=" * 40)
    
    dimensions = [2, 5, 10, 20, 50]
    n_samples = 500
    
    nn_results = []
    
    for d in dimensions:
        # Generate random data
        data = np.random.randn(n_samples, d)
        
        # Fit nearest neighbors
        nn = NearestNeighbors(n_neighbors=2)  # 2 because first is the point itself
        nn.fit(data)
        
        # Find nearest neighbor distances
        distances, indices = nn.kneighbors(data)
        nn_distances = distances[:, 1]  # Skip distance to self (which is 0)
        
        # Statistics
        mean_nn_dist = np.mean(nn_distances)
        std_nn_dist = np.std(nn_distances)
        min_nn_dist = np.min(nn_distances)
        max_nn_dist = np.max(nn_distances)
        
        nn_results.append({
            'dimension': d,
            'mean_nn_distance': mean_nn_dist,
            'std_nn_distance': std_nn_dist,
            'min_nn_distance': min_nn_dist,
            'max_nn_distance': max_nn_dist,
            'nn_distances': nn_distances
        })
        
        print(f"d={d:2d}: mean_nn={mean_nn_dist:.3f}, std_nn={std_nn_dist:.3f}, "
              f"range=[{min_nn_dist:.3f}, {max_nn_dist:.3f}]")
    
    # Plot nearest neighbor distance distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, result in enumerate(nn_results):
        if i < len(axes):
            nn_distances = result['nn_distances']
            
            axes[i].hist(nn_distances, bins=30, alpha=0.7, density=True)
            axes[i].axvline(result['mean_nn_distance'], color='red', linestyle='-',
                           label=f'Mean: {result["mean_nn_distance"]:.3f}')
            axes[i].set_title(f'Dimension {result["dimension"]}')
            axes[i].set_xlabel('Nearest Neighbor Distance')
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # Summary plot
    if len(nn_results) < len(axes):
        ax_summary = axes[len(nn_results)]
        dims = [r['dimension'] for r in nn_results]
        mean_nn_dists = [r['mean_nn_distance'] for r in nn_results]
        std_nn_dists = [r['std_nn_distance'] for r in nn_results]
        
        ax_summary.errorbar(dims, mean_nn_dists, yerr=std_nn_dists, 
                           marker='o', capsize=5, capthick=2)
        ax_summary.set_xlabel('Dimension')
        ax_summary.set_ylabel('Nearest Neighbor Distance')
        ax_summary.set_title('NN Distance vs Dimension')
        ax_summary.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return nn_results

def data_sparsity_analysis():
    """
    Analyze data sparsity in high-dimensional spaces.
    """
    print("\nData Sparsity Analysis")
    print("=" * 30)
    
    dimensions = np.arange(1, 21)
    n_samples = 1000
    
    # For each dimension, calculate the fraction of unit hypercube
    # that contains data points (approximated by nearest neighbor distances)
    
    sparsity_results = []
    
    for d in dimensions:
        # Generate uniform data in [0,1]^d hypercube
        data = np.random.uniform(0, 1, (n_samples, d))
        
        # Estimate local density using k-NN
        nn = NearestNeighbors(n_neighbors=6)  # Use k=5 neighbors
        nn.fit(data)
        distances, _ = nn.kneighbors(data)
        
        # Average distance to 5th nearest neighbor
        avg_5nn_distance = np.mean(distances[:, 5])
        
        # Volume of d-dimensional ball with this radius
        local_volume = hypersphere_volume(d, avg_5nn_distance)
        
        # Fraction of unit hypercube volume
        hypercube_vol = 1.0  # Unit hypercube has volume 1
        density_fraction = local_volume / hypercube_vol
        
        sparsity_results.append({
            'dimension': d,
            'avg_5nn_distance': avg_5nn_distance,
            'local_volume': local_volume,
            'density_fraction': density_fraction
        })
        
        if d <= 10 or d % 5 == 0:
            print(f"d={d:2d}: avg_5nn_dist={avg_5nn_distance:.4f}, "
                  f"density_fraction={density_fraction:.2e}")
    
    # Plot sparsity measures
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    dims = [r['dimension'] for r in sparsity_results]
    avg_distances = [r['avg_5nn_distance'] for r in sparsity_results]
    density_fractions = [r['density_fraction'] for r in sparsity_results]
    
    # Average 5-NN distance
    axes[0].plot(dims, avg_distances, 'o-')
    axes[0].set_xlabel('Dimension')
    axes[0].set_ylabel('Average 5-NN Distance')
    axes[0].set_title('Data Point Separation')
    axes[0].grid(True, alpha=0.3)
    
    # Density fraction (log scale)
    axes[1].semilogy(dims, density_fractions, 'o-', color='red')
    axes[1].set_xlabel('Dimension')
    axes[1].set_ylabel('Local Density Fraction (log)')
    axes[1].set_title('Data Sparsity')
    axes[1].grid(True, alpha=0.3)
    
    # Effective number of neighbors in unit ball
    # (inverse measure of sparsity)
    effective_neighbors = [1.0 / df if df > 0 else np.inf for df in density_fractions]
    axes[2].semilogy(dims, effective_neighbors, 'o-', color='green')
    axes[2].set_xlabel('Dimension')
    axes[2].set_ylabel('Effective Volume per Point (log)')
    axes[2].set_title('Space per Data Point')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return sparsity_results

def density_estimation_failure():
    """
    Demonstrate why density estimation fails in high dimensions.
    """
    print("\nDensity Estimation Failure in High Dimensions")
    print("=" * 55)
    
    dimensions = [2, 5, 10, 15, 20, 30, 50]
    n_samples = 1000
    
    results = []
    
    for d in dimensions:
        print(f"\nDimension {d}:")
        
        # Generate data from mixture of Gaussians
        n_per_cluster = n_samples // 2
        cluster1 = np.random.multivariate_normal(np.zeros(d), np.eye(d), n_per_cluster)
        cluster2 = np.random.multivariate_normal(2*np.ones(d), np.eye(d), n_samples - n_per_cluster)
        data = np.vstack([cluster1, cluster2])
        
        # Calculate cluster separation metrics
        separation_distance = np.linalg.norm(2*np.ones(d))
        relative_separation = separation_distance / np.sqrt(d)
        
        # Try Parzen window estimation with different bandwidths
        from sklearn.neighbors import KernelDensity
        
        # Adaptive bandwidth range based on dimension
        if d <= 5:
            bandwidths = [0.1, 0.3, 0.5, 1.0, 2.0]
        elif d <= 15:
            bandwidths = [0.5, 1.0, 2.0, 3.0, 5.0]
        else:
            bandwidths = [1.0, 2.0, 5.0, 10.0, 20.0]
        
        best_score = -np.inf
        best_score_per_sample = -np.inf
        best_bandwidth = None
        
        for h in bandwidths:
            try:
                kde = KernelDensity(bandwidth=h)
                kde.fit(data)
                
                # Total log-likelihood
                total_score = kde.score(data)
                # Per-sample log-likelihood
                per_sample_score = total_score / n_samples
                
                if per_sample_score > best_score_per_sample:
                    best_score = total_score
                    best_score_per_sample = per_sample_score
                    best_bandwidth = h
                    
            except Exception as e:
                print(f"    Error with bandwidth {h}: {e}")
                continue
        
        # Calculate nearest neighbor distances
        try:
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(data)
            distances, _ = nn.kneighbors(data)
            avg_nn_dist = np.mean(distances[:, 1])  # Distance to nearest neighbor (not self)
        except:
            avg_nn_dist = np.nan
        
        # Theoretical analysis
        min_volume = hypersphere_volume(d, 0.5)  # Arbitrary small radius
        max_theoretical_density = n_samples / min_volume if min_volume > 0 else np.inf
        
        print(f"  Cluster separation distance: {separation_distance:.3f}")
        print(f"  Relative separation (dist/√d): {relative_separation:.3f}")
        print(f"  Best bandwidth: {best_bandwidth}")
        print(f"  Total log-likelihood: {best_score:.3f}")
        print(f"  Per-sample log-likelihood: {best_score_per_sample:.3f}")
        print(f"  Average nearest neighbor distance: {avg_nn_dist:.3f}")
        print(f"  Theoretical max density (r=0.5): {max_theoretical_density:.2e}")
        
        results.append({
            'dimension': d,
            'separation_distance': separation_distance,
            'relative_separation': relative_separation,
            'best_bandwidth': best_bandwidth,
            'total_score': best_score,
            'per_sample_score': best_score_per_sample,
            'avg_nn_distance': avg_nn_dist,
            'theoretical_density': max_theoretical_density
        })
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    dims = [r['dimension'] for r in results]
    total_scores = [r['total_score'] for r in results]
    per_sample_scores = [r['per_sample_score'] for r in results]
    separations = [r['relative_separation'] for r in results]
    nn_distances = [r['avg_nn_distance'] for r in results if not np.isnan(r['avg_nn_distance'])]
    dims_nn = [r['dimension'] for r in results if not np.isnan(r['avg_nn_distance'])]
    bandwidths = [r['best_bandwidth'] for r in results if r['best_bandwidth'] is not None]
    dims_bw = [r['dimension'] for r in results if r['best_bandwidth'] is not None]
    
    # Total log-likelihood (misleading metric)
    axes[0, 0].plot(dims, total_scores, 'o-', color='red', label='Total LL')
    axes[0, 0].set_xlabel('Dimension')
    axes[0, 0].set_ylabel('Total Log-likelihood')
    axes[0, 0].set_title('Total Log-likelihood (Misleading!)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Per-sample log-likelihood (correct metric)
    axes[0, 1].plot(dims, per_sample_scores, 'o-', color='blue', label='Per-sample LL')
    axes[0, 1].set_xlabel('Dimension')
    axes[0, 1].set_ylabel('Log-likelihood per Sample')
    axes[0, 1].set_title('Per-sample Log-likelihood (Correct)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Cluster separation
    axes[0, 2].plot(dims, separations, 'o-', color='green')
    axes[0, 2].set_xlabel('Dimension')
    axes[0, 2].set_ylabel('Relative Separation (dist/√d)')
    axes[0, 2].set_title('Cluster Separability')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Nearest neighbor distances
    if len(dims_nn) > 0:
        axes[1, 0].plot(dims_nn, nn_distances, 'o-', color='purple')
    axes[1, 0].set_xlabel('Dimension')
    axes[1, 0].set_ylabel('Average NN Distance')
    axes[1, 0].set_title('Data Sparsity')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Optimal bandwidths
    if len(dims_bw) > 0:
        axes[1, 1].plot(dims_bw, bandwidths, 'o-', color='orange')
    axes[1, 1].set_xlabel('Dimension')
    axes[1, 1].set_ylabel('Optimal Bandwidth')
    axes[1, 1].set_title('Bandwidth Selection')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Summary comparison
    axes[1, 2].plot(dims, per_sample_scores, 'o-', color='blue', label='LL per sample')
    axes[1, 2].plot(dims, separations, 'o-', color='green', label='Separability')
    axes[1, 2].set_xlabel('Dimension')
    axes[1, 2].set_ylabel('Normalized Metrics')
    axes[1, 2].set_title('Performance Summary')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print analysis
    print(f"\nAnalysis:")
    print(f"- Total log-likelihood can be misleading in high dimensions")
    print(f"- Per-sample log-likelihood shows true performance degradation")
    print(f"- Cluster separation becomes relatively smaller as dimension increases")
    print(f"- Optimal bandwidth must increase dramatically")
    
    return results

def comparison_low_vs_high_dim():
    """
    Direct comparison between low and high dimensional cases.
    """
    print("\nDirect Comparison: Low vs High Dimensions")
    print("=" * 50)
    
    # Generate 2D data (low dimensional)
    np.random.seed(42)
    n_samples = 1000
    
    # 2D case
    data_2d = np.vstack([
        np.random.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], n_samples//2),
        np.random.multivariate_normal([3, 3], [[1, -0.3], [-0.3, 1]], n_samples//2)
    ])
    
    # 20D case (reduced from 50D to avoid computational issues)
    mean1_20d = np.zeros(20)
    mean2_20d = np.zeros(20)
    mean2_20d[:2] = [3, 3]  # Only first two dimensions have the pattern
    
    cov_20d = np.eye(20)
    cov_20d[0, 1] = cov_20d[1, 0] = 0.3
    
    data_20d = np.vstack([
        np.random.multivariate_normal(mean1_20d, cov_20d, n_samples//2),
        np.random.multivariate_normal(mean2_20d, cov_20d, n_samples//2)
    ])
    
    # Compare density estimation performance
    from sklearn.neighbors import KernelDensity
    from sklearn.model_selection import GridSearchCV
    
    print("2D Case:")
    # 2D density estimation
    params = {'bandwidth': np.logspace(-1, 1, 10)}  # Reduced grid size
    
    try:
        grid_2d = GridSearchCV(KernelDensity(), params, cv=3)  # Reduced CV folds
        grid_2d.fit(data_2d)
        
        kde_2d = grid_2d.best_estimator_
        score_2d = kde_2d.score(data_2d)
        
        print(f"  Best bandwidth: {grid_2d.best_params_['bandwidth']:.3f}")
        print(f"  Cross-validated score: {grid_2d.best_score_:.3f}")
        print(f"  Final score: {score_2d:.3f}")
    except Exception as e:
        print(f"  Error in 2D case: {e}")
        # Fallback to simple bandwidth selection
        kde_2d = KernelDensity(bandwidth=1.0)
        kde_2d.fit(data_2d)
        score_2d = kde_2d.score(data_2d)
        print(f"  Fallback bandwidth: 1.0")
        print(f"  Final score: {score_2d:.3f}")
    
    print("\n20D Case:")
    # 20D density estimation
    try:
        grid_20d = GridSearchCV(KernelDensity(), params, cv=3)
        grid_20d.fit(data_20d)
        
        kde_20d = grid_20d.best_estimator_
        score_20d = kde_20d.score(data_20d)
        
        print(f"  Best bandwidth: {grid_20d.best_params_['bandwidth']:.3f}")
        print(f"  Cross-validated score: {grid_20d.best_score_:.3f}")
        print(f"  Final score: {score_20d:.3f}")
    except Exception as e:
        print(f"  Error in 20D case: {e}")
        # Fallback to simple bandwidth selection
        kde_20d = KernelDensity(bandwidth=2.0)
        kde_20d.fit(data_20d)
        score_20d = kde_20d.score(data_20d)
        print(f"  Fallback bandwidth: 2.0")
        print(f"  Final score: {score_20d:.3f}")
    
    # Visualize 2D case
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 2D data
    axes[0].scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.6)
    axes[0].set_title('2D Data')
    axes[0].set_xlabel('X₁')
    axes[0].set_ylabel('X₂')
    axes[0].axis('equal')
    
    # 2D density estimation
    try:
        x_min, x_max = data_2d[:, 0].min()-1, data_2d[:, 0].max()+1
        y_min, y_max = data_2d[:, 1].min()-1, data_2d[:, 1].max()+1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                             np.linspace(y_min, y_max, 50))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        density_2d = np.exp(kde_2d.score_samples(grid_points)).reshape(xx.shape)
        
        contour = axes[1].contour(xx, yy, density_2d, levels=10)
        axes[1].scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.4, s=10)
        axes[1].set_title('2D Density Estimation')
        axes[1].set_xlabel('X₁')
        axes[1].set_ylabel('X₂')
        axes[1].axis('equal')
    except Exception as e:
        axes[1].text(0.5, 0.5, f'Density plot failed:\n{str(e)[:50]}...', 
                     transform=axes[1].transAxes, ha='center', va='center')
        axes[1].set_title('2D Density Estimation (Failed)')
    
    # 20D data projected to first 2 dimensions
    axes[2].scatter(data_20d[:, 0], data_20d[:, 1], alpha=0.6)
    axes[2].set_title('20D Data (projected to first 2D)')
    axes[2].set_xlabel('X₁')
    axes[2].set_ylabel('X₂')
    axes[2].axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nPerformance degradation: {score_2d - score_20d:.3f}")
    print(f"Relative performance: {score_20d/score_2d:.3f}")
    
    return data_2d, data_20d, score_2d, score_20d

def main():
    """Main function demonstrating curse of dimensionality."""
    print("Curse of Dimensionality Demonstration")
    print("=" * 45)
    
    # 1. Volume concentration
    print("1. Hypersphere volume analysis...")
    dimensions, volumes, ratios = analyze_volume_concentration()
    
    # 2. Distance concentration  
    print("\n2. Distance concentration analysis...")
    distance_results = distance_concentration_demo()
    
    # 3. Nearest neighbor analysis
    print("\n3. Nearest neighbor distance analysis...")
    nn_results = nearest_neighbor_analysis()
    
    # 4. Data sparsity
    print("\n4. Data sparsity analysis...")
    sparsity_results = data_sparsity_analysis()
    
    # 5. Density estimation failure
    print("\n5. Density estimation in high dimensions...")
    density_estimation_failure()
    
    # 6. Direct comparison
    print("\n6. Low vs high dimensional comparison...")
    comparison_low_vs_high_dim()
    
    # Summary
    print(f"\nSummary of Curse of Dimensionality Effects:")
    print(f"- Volume concentration: peaks around d=5, then decays exponentially")
    print(f"- Distance concentration: relative std decreases as 1/√(2d)")
    print(f"- Data sparsity: exponentially increasing space per data point")
    print(f"- Density estimation: becomes unreliable for d > 10")
    print(f"- Nearest neighbors: distances become similar, losing discriminative power")

if __name__ == "__main__":
    main()