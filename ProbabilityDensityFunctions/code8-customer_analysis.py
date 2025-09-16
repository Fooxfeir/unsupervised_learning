"""
Complete Customer Behavior Analysis
===================================

This module provides a comprehensive analysis of customer behavior data
using all PDF estimation methods covered in the lecture.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy.stats import multivariate_normal
import seaborn as sns

# Note: Import custom implementations if available, otherwise use alternatives
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from em_algorithm import GaussianMixtureEM
    CUSTOM_EM_AVAILABLE = True
    print("✓ Custom EM implementation loaded successfully")
except (ImportError, SyntaxError) as e:
    print(f"⚠ Custom EM implementation not available: {e}")
    CUSTOM_EM_AVAILABLE = False

try:
    from kl_divergence import gaussian_kl_divergence, gmm_kl_divergence_mc
    CUSTOM_KL_AVAILABLE = True
    print("✓ Custom KL divergence implementation loaded successfully")
except (ImportError, SyntaxError) as e:
    print(f"⚠ Custom KL divergence implementation not available: {e}")
    CUSTOM_KL_AVAILABLE = False

try:
    from whitening_transform import compute_whitening_matrix, apply_whitening
    CUSTOM_WHITENING_AVAILABLE = True
    print("✓ Custom whitening implementation loaded successfully")
except (ImportError, SyntaxError) as e:
    print(f"⚠ Custom whitening implementation not available: {e}")
    CUSTOM_WHITENING_AVAILABLE = False

def simple_whitening_transform(data):
    """
    Simple whitening transform implementation as fallback.
    """
    # Center the data
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    
    # Compute covariance matrix
    cov = np.cov(centered_data.T)
    
    # Eigendecomposition
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    
    # Whitening matrix
    epsilon = 1e-8
    W = eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals + epsilon)) @ eigenvecs.T
    
    # Apply whitening
    whitened_data = (W @ centered_data.T).T
    
    return whitened_data, W, mean

def generate_customer_data():
    """
    Generate realistic customer behavior data.
    
    Returns:
    --------
    data : array, shape (n_samples, 2)
        Customer data [monthly_spending, visit_frequency]
    true_labels : array, shape (n_samples,)
        True customer segment labels
    segment_info : dict
        Information about each customer segment
    """
    np.random.seed(42)
    
    # Segment 1: Regular customers
    # Moderate spending, consistent visits
    n1 = 200
    mean1 = np.array([50, 10])  # $50/month, 10 visits/month
    cov1 = np.array([[100, 20], [20, 25]])  # Some correlation
    cluster1 = np.random.multivariate_normal(mean1, cov1, n1)
    
    # Segment 2: Premium customers  
    # High spending, frequent visits, some anti-correlation
    n2 = 150
    mean2 = np.array([120, 25])  # $120/month, 25 visits/month
    cov2 = np.array([[200, -30], [-30, 40]])  # Negative correlation
    cluster2 = np.random.multivariate_normal(mean2, cov2, n2)
    
    # Segment 3: Occasional customers
    # Low spending, infrequent visits
    n3 = 100
    mean3 = np.array([30, 5])  # $30/month, 5 visits/month
    cov3 = np.array([[50, 10], [10, 15]])  # Low variance
    cluster3 = np.random.multivariate_normal(mean3, cov3, n3)
    
    # Combine data
    data = np.vstack([cluster1, cluster2, cluster3])
    true_labels = np.hstack([np.zeros(n1), np.ones(n2), np.full(n3, 2)])
    
    # Ensure no negative values (spending and frequency can't be negative)
    data = np.maximum(data, 0.1)
    
    segment_info = {
        0: {'name': 'Regular', 'n': n1, 'mean': mean1, 'cov': cov1},
        1: {'name': 'Premium', 'n': n2, 'mean': mean2, 'cov': cov2},
        2: {'name': 'Occasional', 'n': n3, 'mean': mean3, 'cov': cov3}
    }
    
    return data, true_labels, segment_info

def parzen_window_analysis(data):
    """
    Perform Parzen window analysis with bandwidth selection.
    
    Parameters:
    -----------
    data : array-like
        Customer data
    
    Returns:
    --------
    kde : KernelDensity
        Fitted KDE model
    best_bandwidth : float
        Optimal bandwidth
    """
    print("Parzen Window Analysis")
    print("=" * 25)
    
    # Try different bandwidths with manual cross-validation
    from sklearn.model_selection import cross_val_score
    
    bandwidths = np.logspace(-1, 1.5, 20)
    scores = []
    
    for bw in bandwidths:
        kde = KernelDensity(bandwidth=bw, kernel='gaussian')
        # Use cross-validation to get more robust bandwidth selection
        cv_scores = cross_val_score(kde, data, cv=3, scoring='neg_log_loss')
        mean_score = np.mean(cv_scores)
        scores.append(mean_score)
    
    # Select best bandwidth
    best_idx = np.argmax(scores)
    best_bandwidth = bandwidths[best_idx]
    best_score = scores[best_idx]
    
    print(f"Best bandwidth: {best_bandwidth:.3f}")
    print(f"Best cross-validated score: {best_score:.3f}")
    
    # Fit final model with best bandwidth
    kde = KernelDensity(bandwidth=best_bandwidth, kernel='gaussian')
    kde.fit(data)
    
    # Compute final log-likelihood
    final_score = kde.score(data)
    print(f"Final log-likelihood: {final_score:.3f}")
    
    # Plot bandwidth selection
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.semilogx(bandwidths, scores, 'o-')
    plt.axvline(best_bandwidth, color='red', linestyle='--', 
                label=f'Best: {best_bandwidth:.3f}')
    plt.xlabel('Bandwidth')
    plt.ylabel('Cross-validated Score')
    plt.title('Bandwidth Selection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot data with selected bandwidth
    plt.subplot(1, 2, 2)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.6)
    plt.xlabel('Monthly Spending ($)')
    plt.ylabel('Visit Frequency')
    plt.title(f'Customer Data (h={best_bandwidth:.3f})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return kde, best_bandwidth

def gaussian_mixture_analysis(data, true_labels):
    """
    Fit Gaussian mixture models with different numbers of components.
    
    Parameters:
    -----------
    data : array-like
        Customer data
    true_labels : array-like
        True segment labels
    
    Returns:
    --------
    best_gmm : GaussianMixture
        Best fitted GMM model
    """
    print("\nGaussian Mixture Analysis")
    print("=" * 30)
    
    # Try different numbers of components
    n_components_range = range(1, 8)
    aic_scores = []
    bic_scores = []
    log_likelihoods = []
    
    models = {}
    
    for n_comp in n_components_range:
        gmm = GaussianMixture(n_components=n_comp, random_state=42, max_iter=200)
        gmm.fit(data)
        
        aic = gmm.aic(data)
        bic = gmm.bic(data)
        ll = gmm.score(data) * len(data)  # Total log-likelihood
        
        aic_scores.append(aic)
        bic_scores.append(bic)
        log_likelihoods.append(ll)
        models[n_comp] = gmm
        
        print(f"Components: {n_comp}, AIC: {aic:.1f}, BIC: {bic:.1f}, LL: {ll:.1f}")
    
    # Select best model based on BIC
    best_n_comp = n_components_range[np.argmin(bic_scores)]
    best_gmm = models[best_n_comp]
    
    print(f"\nBest model: {best_n_comp} components (lowest BIC)")
    
    # Plot model selection criteria
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(n_components_range, aic_scores, 'o-', label='AIC')
    plt.plot(n_components_range, bic_scores, 's-', label='BIC')
    plt.axvline(best_n_comp, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Number of Components')
    plt.ylabel('Information Criterion')
    plt.title('Model Selection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(n_components_range, log_likelihoods, 'o-', color='green')
    plt.xlabel('Number of Components')
    plt.ylabel('Log-likelihood')
    plt.title('Model Fit')
    plt.grid(True, alpha=0.3)
    
    # Compare with our EM implementation if available
    plt.subplot(1, 3, 3)
    if CUSTOM_EM_AVAILABLE:
        our_gmm = GaussianMixtureEM(n_components=best_n_comp, random_state=42)
        our_gmm.fit(data)
        
        sklearn_ll = best_gmm.score_samples(data).mean()
        our_ll = our_gmm.score_samples(data).mean()
        
        plt.bar(['sklearn GMM', 'Our EM'], [sklearn_ll, our_ll])
        plt.ylabel('Mean Log-likelihood')
        plt.title('Implementation Comparison')
    else:
        # Just show sklearn results
        sklearn_ll = best_gmm.score_samples(data).mean()
        plt.bar(['sklearn GMM'], [sklearn_ll])
        plt.ylabel('Mean Log-likelihood')
        plt.title('GMM Log-likelihood')
        plt.text(0.5, 0.5, 'Custom EM\nNot Available', transform=plt.gca().transAxes,
                ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightgray'))
    
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return best_gmm

def whitening_analysis(data):
    """
    Demonstrate whitening transform effects.
    
    Parameters:
    -----------
    data : array-like
        Customer data
    
    Returns:
    --------
    whitened_data : array-like
        Whitened data
    W : array-like
        Whitening matrix
    data_mean : array-like
        Original data mean
    """
    print("\nWhitening Transform Analysis")
    print("=" * 35)
    
    # Compute original statistics
    original_mean = np.mean(data, axis=0)
    original_cov = np.cov(data.T)
    original_corr = np.corrcoef(data.T)
    
    print("Original data statistics:")
    print(f"Mean: [{original_mean[0]:.1f}, {original_mean[1]:.1f}]")
    print(f"Covariance:\n{original_cov}")
    print(f"Correlation:\n{original_corr}")
    
    # Apply whitening transform
    if CUSTOM_WHITENING_AVAILABLE:
        W, data_mean = compute_whitening_matrix(data, method='ZCA')
        whitened_data = apply_whitening(data, W, data_mean)
    else:
        whitened_data, W, data_mean = simple_whitening_transform(data)
    
    # Compute whitened statistics
    whitened_mean = np.mean(whitened_data, axis=0)
    whitened_cov = np.cov(whitened_data.T)
    whitened_corr = np.corrcoef(whitened_data.T)
    
    print(f"\nWhitened data statistics:")
    print(f"Mean: [{whitened_mean[0]:.3f}, {whitened_mean[1]:.3f}]")
    print(f"Covariance:\n{whitened_cov}")
    print(f"Correlation:\n{whitened_corr}")
    print(f"Is identity covariance? {np.allclose(whitened_cov, np.eye(2), atol=1e-6)}")
    
    # Visualize transformation
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original data
    axes[0, 0].scatter(data[:, 0], data[:, 1], alpha=0.6)
    axes[0, 0].set_title('Original Data')
    axes[0, 0].set_xlabel('Monthly Spending ($)')
    axes[0, 0].set_ylabel('Visit Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Original covariance ellipse
    from matplotlib.patches import Ellipse
    eigenvals, eigenvecs = np.linalg.eigh(original_cov)
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    width = 2 * 2 * np.sqrt(eigenvals[0])  # 2 std devs
    height = 2 * 2 * np.sqrt(eigenvals[1])
    ellipse = Ellipse(original_mean, width, height, angle=angle, 
                     fill=False, edgecolor='red', linewidth=2)
    axes[0, 0].add_patch(ellipse)
    
    # Whitened data
    axes[0, 1].scatter(whitened_data[:, 0], whitened_data[:, 1], alpha=0.6)
    axes[0, 1].set_title('Whitened Data')
    axes[0, 1].set_xlabel('Whitened Dimension 1')
    axes[0, 1].set_ylabel('Whitened Dimension 2')
    axes[0, 1].axis('equal')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add unit circles for reference
    circle1 = plt.Circle((0, 0), 1, fill=False, edgecolor='red', linewidth=2, linestyle='--')
    circle2 = plt.Circle((0, 0), 2, fill=False, edgecolor='red', linewidth=2, linestyle='--')
    axes[0, 1].add_patch(circle1)
    axes[0, 1].add_patch(circle2)
    
    # Covariance matrices comparison
    im1 = axes[0, 2].imshow(original_cov, cmap='coolwarm', aspect='equal')
    axes[0, 2].set_title('Original Covariance')
    axes[0, 2].set_xticks([0, 1])
    axes[0, 2].set_yticks([0, 1])
    axes[0, 2].set_xticklabels(['Spending', 'Frequency'])
    axes[0, 2].set_yticklabels(['Spending', 'Frequency'])
    plt.colorbar(im1, ax=axes[0, 2])
    
    im2 = axes[1, 0].imshow(whitened_cov, cmap='coolwarm', aspect='equal', vmin=-1, vmax=1)
    axes[1, 0].set_title('Whitened Covariance')
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_xticklabels(['Dim 1', 'Dim 2'])
    axes[1, 0].set_yticklabels(['Dim 1', 'Dim 2'])
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Joint distribution comparison
    # Create grid for PDF evaluation
    x_range = np.linspace(data[:, 0].min()-20, data[:, 0].max()+20, 60)
    y_range = np.linspace(data[:, 1].min()-10, data[:, 1].max()+10, 60)
    X, Y = np.meshgrid(x_range, y_range)
    pos_orig = np.dstack((X, Y))
    
    # Original space PDF
    rv_orig = multivariate_normal(original_mean, original_cov)
    pdf_orig = rv_orig.pdf(pos_orig)
    
    contour1 = axes[1, 1].contour(X, Y, pdf_orig, levels=8)
    axes[1, 1].scatter(data[:, 0], data[:, 1], alpha=0.4, s=15)
    axes[1, 1].set_title('Original Space PDF')
    axes[1, 1].set_xlabel('Monthly Spending ($)')
    axes[1, 1].set_ylabel('Visit Frequency')
    
    # Whitened space PDF
    z_range = np.linspace(whitened_data[:, 0].min()-2, whitened_data[:, 0].max()+2, 60)
    w_range = np.linspace(whitened_data[:, 1].min()-2, whitened_data[:, 1].max()+2, 60)
    Z, W_mesh = np.meshgrid(z_range, w_range)
    pos_white = np.dstack((Z, W_mesh))
    
    rv_white = multivariate_normal([0, 0], np.eye(2))
    pdf_white = rv_white.pdf(pos_white)
    
    contour2 = axes[1, 2].contour(Z, W_mesh, pdf_white, levels=8)
    axes[1, 2].scatter(whitened_data[:, 0], whitened_data[:, 1], alpha=0.4, s=15)
    axes[1, 2].set_title('Whitened Space PDF')
    axes[1, 2].set_xlabel('Whitened Dimension 1')
    axes[1, 2].set_ylabel('Whitened Dimension 2')
    axes[1, 2].axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    return whitened_data, W, data_mean

def comprehensive_comparison(data, true_labels):
    """
    Compare all PDF estimation methods comprehensively.
    
    Parameters:
    -----------
    data : array-like
        Customer data
    true_labels : array-like
        True segment labels
    
    Returns:
    --------
    results : dict
        Comparison results
    """
    print("\nComprehensive Method Comparison")
    print("=" * 40)
    
    # 1. Parzen Window
    kde, best_bandwidth = parzen_window_analysis(data)
    
    # 2. Gaussian Mixture
    gmm = gaussian_mixture_analysis(data, true_labels)
    
    # 3. Single Gaussian (for comparison)
    single_gaussian_mean = np.mean(data, axis=0)
    single_gaussian_cov = np.cov(data.T)
    single_gaussian = multivariate_normal(single_gaussian_mean, single_gaussian_cov)
    
    # Create evaluation grid
    x_min, x_max = data[:, 0].min() - 20, data[:, 0].max() + 20
    y_min, y_max = data[:, 1].min() - 10, data[:, 1].max() + 10
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Evaluate all methods on grid
    kde_density = np.exp(kde.score_samples(grid_points)).reshape(xx.shape)
    gmm_density = np.exp(gmm.score_samples(grid_points)).reshape(xx.shape)
    single_density = single_gaussian.pdf(grid_points.reshape(-1, 1, 2)).squeeze().reshape(xx.shape)
    
    # Compute log-likelihoods on data
    kde_ll = np.mean(kde.score_samples(data))
    gmm_ll = np.mean(gmm.score_samples(data))
    single_ll = np.mean(single_gaussian.logpdf(data))
    
    print(f"\nLog-likelihood comparison:")
    print(f"Parzen Window: {kde_ll:.4f}")
    print(f"Mixture of Gaussians: {gmm_ll:.4f}")
    print(f"Single Gaussian: {single_ll:.4f}")
    
    # Compute KL divergences (approximate)
    try:
        if CUSTOM_KL_AVAILABLE:
            kl_kde_gmm = gmm_kl_divergence_mc(gmm, kde, n_samples=2000)
            print(f"\nKL(GMM||KDE) ≈ {kl_kde_gmm:.4f}")
    except:
        print(f"\nKL divergence computation failed (different interfaces)")
    
    # Create separate figures for better organization
    # Figure 1: 2D Contour plots
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original data with true clusters
    scatter = axes1[0, 0].scatter(data[:, 0], data[:, 1], c=true_labels, alpha=0.7, cmap='tab10')
    axes1[0, 0].set_title('True Customer Segments')
    axes1[0, 0].set_xlabel('Monthly Spending ($)')
    axes1[0, 0].set_ylabel('Visit Frequency')
    plt.colorbar(scatter, ax=axes1[0, 0])
    
    # Parzen window
    contour1 = axes1[0, 1].contour(xx, yy, kde_density, levels=10, alpha=0.8)
    axes1[0, 1].scatter(data[:, 0], data[:, 1], alpha=0.4, s=15)
    axes1[0, 1].set_title(f'Parzen Window (h={best_bandwidth:.3f})')
    axes1[0, 1].set_xlabel('Monthly Spending ($)')
    axes1[0, 1].set_ylabel('Visit Frequency')
    
    # Mixture of Gaussians
    contour2 = axes1[1, 0].contour(xx, yy, gmm_density, levels=10, alpha=0.8)
    axes1[1, 0].scatter(data[:, 0], data[:, 1], alpha=0.4, s=15)
    # Plot component means
    for k in range(gmm.n_components):
        axes1[1, 0].scatter(gmm.means_[k, 0], gmm.means_[k, 1],
                          marker='x', s=200, linewidth=3, color='red')
    axes1[1, 0].set_title(f'Mixture of Gaussians ({gmm.n_components} comp)')
    axes1[1, 0].set_xlabel('Monthly Spending ($)')
    axes1[1, 0].set_ylabel('Visit Frequency')
    
    # Single Gaussian
    contour3 = axes1[1, 1].contour(xx, yy, single_density, levels=10, alpha=0.8)
    axes1[1, 1].scatter(data[:, 0], data[:, 1], alpha=0.4, s=15)
    axes1[1, 1].set_title('Single Gaussian')
    axes1[1, 1].set_xlabel('Monthly Spending ($)')
    axes1[1, 1].set_ylabel('Visit Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # Figure 2: 3D Surface plots
    fig2 = plt.figure(figsize=(15, 5))
    
    # Create smaller grid for 3D plotting (for performance)
    X_small, Y_small = np.meshgrid(np.linspace(x_min, x_max, 50),
                                   np.linspace(y_min, y_max, 50))
    grid_small = np.c_[X_small.ravel(), Y_small.ravel()]
    
    # Parzen window 3D
    ax_3d1 = fig2.add_subplot(131, projection='3d')
    Z_kde = np.exp(kde.score_samples(grid_small)).reshape(X_small.shape)
    surf1 = ax_3d1.plot_surface(X_small, Y_small, Z_kde, cmap='viridis', alpha=0.8)
    ax_3d1.set_title('Parzen Window 3D')
    ax_3d1.set_xlabel('Spending ($)')
    ax_3d1.set_ylabel('Frequency')
    ax_3d1.set_zlabel('Density')
    
    # GMM 3D
    ax_3d2 = fig2.add_subplot(132, projection='3d')
    Z_gmm = np.exp(gmm.score_samples(grid_small)).reshape(X_small.shape)
    surf2 = ax_3d2.plot_surface(X_small, Y_small, Z_gmm, cmap='plasma', alpha=0.8)
    ax_3d2.set_title('GMM 3D')
    ax_3d2.set_xlabel('Spending ($)')
    ax_3d2.set_ylabel('Frequency')
    ax_3d2.set_zlabel('Density')
    
    # Single Gaussian 3D
    ax_3d3 = fig2.add_subplot(133, projection='3d')
    Z_single = single_gaussian.pdf(grid_small.reshape(-1, 1, 2)).squeeze().reshape(X_small.shape)
    surf3 = ax_3d3.plot_surface(X_small, Y_small, Z_single, cmap='coolwarm', alpha=0.8)
    ax_3d3.set_title('Single Gaussian 3D')
    ax_3d3.set_xlabel('Spending ($)')
    ax_3d3.set_ylabel('Frequency')
    ax_3d3.set_zlabel('Density')
    
    plt.tight_layout()
    plt.show()
    
    # Figure 3: Analysis and comparison plots
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
    
    # Log-likelihood comparison
    methods = ['Parzen\nWindow', 'Gaussian\nMixture', 'Single\nGaussian']
    ll_values = [kde_ll, gmm_ll, single_ll]
    
    bars = axes3[0].bar(methods, ll_values, color=['blue', 'green', 'red'], alpha=0.7)
    axes3[0].set_ylabel('Mean Log-likelihood')
    axes3[0].set_title('Model Comparison')
    axes3[0].grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, val in zip(bars, ll_values):
        height = bar.get_height()
        axes3[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{val:.3f}', ha='center', va='bottom')
    
    # Predicted clusters from GMM
    predicted_labels = gmm.predict(data)
    scatter2 = axes3[1].scatter(data[:, 0], data[:, 1], c=predicted_labels, alpha=0.7, cmap='tab10')
    axes3[1].set_title('GMM Predicted Clusters')
    axes3[1].set_xlabel('Monthly Spending ($)')
    axes3[1].set_ylabel('Visit Frequency')
    plt.colorbar(scatter2, ax=axes3[1])
    
    # Model parameters summary
    axes3[2].axis('off')
    param_text = "Model Parameters:\n\n"
    
    param_text += f"Parzen Window:\n"
    param_text += f"  Bandwidth: {best_bandwidth:.3f}\n"
    param_text += f"  Log-likelihood: {kde_ll:.4f}\n\n"
    
    param_text += f"Gaussian Mixture:\n"
    param_text += f"  Components: {gmm.n_components}\n"
    param_text += f"  Log-likelihood: {gmm_ll:.4f}\n"
    for k in range(gmm.n_components):
        param_text += f"  Component {k+1}:\n"
        param_text += f"    Weight: {gmm.weights_[k]:.3f}\n"
        param_text += f"    Mean: [{gmm.means_[k, 0]:.1f}, {gmm.means_[k, 1]:.1f}]\n"
    
    param_text += f"\nSingle Gaussian:\n"
    param_text += f"  Mean: [{single_gaussian_mean[0]:.1f}, {single_gaussian_mean[1]:.1f}]\n"
    param_text += f"  Log-likelihood: {single_ll:.4f}"
    
    axes3[2].text(0.05, 0.95, param_text, transform=axes3[2].transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()
    
    # Performance metrics
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    
    ari_gmm = adjusted_rand_score(true_labels, predicted_labels)
    silhouette_gmm = silhouette_score(data, predicted_labels)
    
    print(f"\nClustering Performance (GMM):")
    print(f"Adjusted Rand Index: {ari_gmm:.3f}")
    print(f"Silhouette Score: {silhouette_gmm:.3f}")
    
    results = {
        'kde': kde,
        'gmm': gmm,
        'single_gaussian': single_gaussian,
        'log_likelihoods': {'kde': kde_ll, 'gmm': gmm_ll, 'single': single_ll},
        'clustering_metrics': {'ari': ari_gmm, 'silhouette': silhouette_gmm},
        'best_bandwidth': best_bandwidth
    }
    
    return results

def business_insights(data, true_labels, segment_info, results):
    """
    Generate business insights from the analysis.
    
    Parameters:
    -----------
    data : array-like
        Customer data
    true_labels : array-like
        True segment labels
    segment_info : dict
        Segment information
    results : dict
        Analysis results
    """
    print("\nBusiness Insights from PDF Analysis")
    print("=" * 45)
    
    gmm = results['gmm']
    
    # Analyze customer segments
    print("Customer Segment Analysis:")
    print("-" * 30)
    
    for k in range(gmm.n_components):
        weight = gmm.weights_[k]
        mean = gmm.means_[k]
        cov = gmm.covariances_[k]
        
        # Calculate confidence intervals
        std_spending = np.sqrt(cov[0, 0])
        std_frequency = np.sqrt(cov[1, 1])
        correlation = cov[0, 1] / (std_spending * std_frequency)
        
        print(f"\nComponent {k+1} ({weight:.1%} of customers):")
        print(f"  Average monthly spending: ${mean[0]:.1f} ± ${1.96*std_spending:.1f}")
        print(f"  Average visit frequency: {mean[1]:.1f} ± {1.96*std_frequency:.1f}")
        print(f"  Spending-frequency correlation: {correlation:.3f}")
        
        # Business interpretation
        if mean[0] > 80 and mean[1] > 20:
            print(f"  → High-value customers (premium segment)")
        elif mean[0] < 40 and mean[1] < 8:
            print(f"  → Low-engagement customers (at-risk segment)")
        else:
            print(f"  → Regular customers (core segment)")
    
    # Revenue analysis
    total_customers = len(data)
    monthly_revenue_per_customer = np.mean(data[:, 0])
    total_monthly_revenue = total_customers * monthly_revenue_per_customer
    
    print(f"\nRevenue Analysis:")
    print(f"  Total customers: {total_customers:,}")
    print(f"  Average monthly revenue per customer: ${monthly_revenue_per_customer:.2f}")
    print(f"  Estimated total monthly revenue: ${total_monthly_revenue:,.2f}")
    
    # Segment-specific strategies
    print(f"\nRecommended Strategies:")
    print("-" * 25)
    
    responsibilities = gmm.predict_proba(data)
    
    for k in range(gmm.n_components):
        segment_customers = np.sum(responsibilities[:, k] > 0.5)
        segment_revenue = np.sum(data[responsibilities[:, k] > 0.5, 0]) if segment_customers > 0 else 0
        
        print(f"\nComponent {k+1} ({segment_customers} customers, ${segment_revenue:,.0f} monthly revenue):")
        
        mean = gmm.means_[k]
        if mean[0] > 80 and mean[1] > 20:
            print(f"  - Implement VIP program")
            print(f"  - Offer premium services")
            print(f"  - Focus on retention")
        elif mean[0] < 40 and mean[1] < 8:
            print(f"  - Re-engagement campaigns")
            print(f"  - Incentive programs")
            print(f"  - Investigate churn factors")
        else:
            print(f"  - Upselling opportunities")
            print(f"  - Loyalty programs")
            print(f"  - Cross-selling campaigns")

def main():
    """Main function for comprehensive customer analysis."""
    print("Complete Customer Behavior Analysis")
    print("=" * 45)
    print("Lecture demonstration: Joint PDF estimation methods")
    print("=" * 45)
    
    # Generate customer data
    print("1. Generating customer behavior data...")
    data, true_labels, segment_info = generate_customer_data()
    
    print(f"Generated {len(data)} customer records")
    print(f"Features: Monthly spending ($), Visit frequency (visits/month)")
    print(f"True segments: {len(segment_info)}")
    
    # Display basic statistics
    print(f"\nData Summary:")
    print(f"Spending range: [${data[:, 0].min():.1f}, ${data[:, 0].max():.1f}]")
    print(f"Frequency range: [{data[:, 1].min():.1f}, {data[:, 1].max():.1f}] visits/month")
    print(f"Correlation: {np.corrcoef(data.T)[0, 1]:.3f}")
    
    # 2. Whitening analysis
    print("\n2. Whitening transform analysis...")
    whitened_data, W, data_mean = whitening_analysis(data)
    
    # 3. Comprehensive comparison
    print("\n3. Comprehensive method comparison...")
    results = comprehensive_comparison(data, true_labels)
    
    # 4. Business insights
    print("\n4. Generating business insights...")
    business_insights(data, true_labels, segment_info, results)
    
    # 5. Final summary
    print(f"\n" + "="*60)
    print("LECTURE SUMMARY: PDF ESTIMATION METHODS")
    print("="*60)
    print(f"✓ Non-parametric (Parzen): Flexible, captures local variations")
    print(f"✓ Parametric (GMM): Interpretable, efficient, good for clustering")
    print(f"✓ Whitening transform: Decorrelates data, simplifies computations")
    print(f"✓ Model comparison: Log-likelihood, AIC/BIC for selection")
    print(f"✓ Real-world application: Customer segmentation and strategy")
    print("="*60)
    
    return data, true_labels, results

if __name__ == "__main__":
    main()
