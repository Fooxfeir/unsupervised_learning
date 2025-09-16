"""Whitening Transform Implementation
=================================

This module demonstrates the whitening transformation for
decorrelating data. It shows three types of whitening transforms.

The primary differences among ZCA, PCA, and Cholesky whitening lie in
their transformation matrices and the resulting properties of the
whitened data, despite all achieving a covariance matrix equal to the
identity matrix.

PCA whitening involves centering the data, computing the covariance
matrix, and performing an eigen-decomposition to obtain eigenvectors
and eigenvalues.  The data is then transformed using a matrix that
rotates the data to the principal component space and scales the
components by the inverse square root of their eigenvalues. This
process decorrelates the features and rescales them to have unit
variance, but the resulting data can have any orientation, which is
optimal for data compression since principal components are sorted by
explained variance.  The transformation matrix for PCA whitening is
derived from the eigenvectors and the inverse square root of the
eigenvalues.

ZCA whitening also begins with centering the data and computing the
covariance matrix.  It uses the same eigen-decomposition as PCA
whitening.  However, the key difference is an additional rotation
step: the PCA-whitened data is rotated back using the eigenvector
matrix.  This results in a transformation matrix that is the product
of the inverse square root of the eigenvalues and the eigenvectors.
The defining property of ZCA whitening is that it minimizes the mean
squared difference between the original and whitened data, meaning the
whitened data is as close as possible to the original data in a
least-squares sense.  This preserves the spatial structure and
orientation of the original data, making it particularly useful for
image processing where local features are important. It is also known
as "zero-phase component analysis" because it minimally distorts the
original phase (orientation) of the data.

Cholesky whitening differs fundamentally by using the Cholesky
decomposition of the inverse covariance matrix to compute a lower
triangular whitening matrix.  This method depends on the ordering of
the input variables, meaning the transformation is not invariant to
variable reordering.  The resulting whitening matrix is lower
triangular with positive diagonal elements, which leads to a specific
cross-covariance and cross-correlation structure between the original
and whitened variables.  This method is unique in that it produces a
lower triangular positive diagonal cross-covariance matrix.

In summary, PCA whitening is optimal for compression, ZCA whitening is
optimal for preserving the original data's structure and orientation,
and Cholesky whitening is unique due to its dependence on variable
ordering and its lower triangular structure

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns

def compute_whitening_matrix(data, method='ZCA'):
    """
    Compute whitening matrix for the given data.
    
    Parameters:
    -----------
    data : array-like, shape (n_samples, n_features)
        Input data
    method : str
        Whitening method ('ZCA', 'PCA', 'Cholesky')
    
    Returns:
    --------
    W : array, shape (n_features, n_features)
        Whitening matrix
    mean : array, shape (n_features,)
        Data mean
    """
    # Center the data
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    
    # Compute covariance matrix
    cov = np.cov(centered_data.T)
    
    if method == 'ZCA':
        # ZCA whitening (zero-phase component analysis)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        # Add small epsilon for numerical stability
        epsilon = 1e-8
        W = eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals + epsilon)) @ eigenvecs.T
        
    elif method == 'PCA':
        # PCA whitening
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        epsilon = 1e-8
        W = np.diag(1.0 / np.sqrt(eigenvals + epsilon)) @ eigenvecs.T
        
    elif method == 'Cholesky':
        # Cholesky whitening
        try:
            L = np.linalg.cholesky(cov)
            W = np.linalg.inv(L)
        except np.linalg.LinAlgError:
            # Fallback to eigendecomposition if Cholesky fails
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            epsilon = 1e-8
            W = eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals + epsilon)) @ eigenvecs.T
    
    else:
        raise ValueError(f"Unknown whitening method: {method}")
    
    return W, mean

def apply_whitening(data, W, mean):
    """
    Apply whitening transformation to data.
    
    Parameters:
    -----------
    data : array-like, shape (n_samples, n_features)
        Input data
    W : array, shape (n_features, n_features)
        Whitening matrix
    mean : array, shape (n_features,)
        Data mean
    
    Returns:
    --------
    whitened_data : array, shape (n_samples, n_features)
        Whitened data
    """
    centered_data = data - mean
    whitened_data = (W @ centered_data.T).T
    return whitened_data

def inverse_whitening(whitened_data, W, mean):
    """
    Inverse whitening transformation.
    
    Parameters:
    -----------
    whitened_data : array-like, shape (n_samples, n_features)
        Whitened data
    W : array, shape (n_features, n_features)
        Whitening matrix
    mean : array, shape (n_features,)
        Original data mean
    
    Returns:
    --------
    reconstructed_data : array, shape (n_samples, n_features)
        Reconstructed original data
    """
    W_inv = np.linalg.inv(W)
    reconstructed_centered = (W_inv @ whitened_data.T).T
    reconstructed_data = reconstructed_centered + mean
    return reconstructed_data

def demonstrate_whitening_methods():
    """
    Compare different whitening methods.
    """
    # Generate correlated 2D data
    np.random.seed(42)
    mean = np.array([2, 1])
    cov = np.array([[2, 1.5], [1.5, 2]])
    data = np.random.multivariate_normal(mean, cov, 300)
    
    methods = ['ZCA', 'PCA', 'Cholesky']
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Plot original data
    axes[0, 0].scatter(data[:, 0], data[:, 1], alpha=0.6)
    axes[0, 0].set_title('Original Data')
    axes[0, 0].set_xlabel('X₁')
    axes[0, 0].set_ylabel('X₂')
    axes[0, 0].axis('equal')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Show covariance ellipse
    from matplotlib.patches import Ellipse
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    width = 2 * 2 * np.sqrt(eigenvals[0])  # 2 standard deviations
    height = 2 * 2 * np.sqrt(eigenvals[1])
    ellipse = Ellipse(mean, width, height, angle=angle, fill=False, 
                     edgecolor='red', linewidth=2)
    axes[0, 0].add_patch(ellipse)
    
    print("Whitening Methods Comparison")
    print("=" * 35)
    print(f"Original covariance:\n{np.cov(data.T)}")
    print(f"Original correlation:\n{np.corrcoef(data.T)}")
    
    # Apply different whitening methods
    for i, method in enumerate(methods):
        W, data_mean = compute_whitening_matrix(data, method)
        whitened = apply_whitening(data, W, data_mean)
        
        # Plot whitened data
        axes[0, i+1].scatter(whitened[:, 0], whitened[:, 1], alpha=0.6)
        axes[0, i+1].set_title(f'{method} Whitening')
        axes[0, i+1].set_xlabel('Z₁')
        axes[0, i+1].set_ylabel('Z₂')
        axes[0, i+1].axis('equal')
        axes[0, i+1].grid(True, alpha=0.3)
        
        # Add unit circle for reference
        circle = plt.Circle((0, 0), 2, fill=False, edgecolor='red', linewidth=2)
        axes[0, i+1].add_patch(circle)
        
        # Verify whitening
        whitened_cov = np.cov(whitened.T)
        print(f"\n{method} whitening:")
        print(f"Whitened covariance:\n{whitened_cov}")
        print(f"Is identity? {np.allclose(whitened_cov, np.eye(2), atol=1e-6)}")
        
        # Reconstruct data
        reconstructed = inverse_whitening(whitened, W, data_mean)
        reconstruction_error = np.mean(np.linalg.norm(data - reconstructed, axis=1))
        print(f"Reconstruction error: {reconstruction_error:.8f}")
        
        # Plot reconstruction
        axes[1, i+1].scatter(reconstructed[:, 0], reconstructed[:, 1], alpha=0.6)
        axes[1, i+1].set_title(f'{method} Reconstructed')
        axes[1, i+1].set_xlabel('X₁')
        axes[1, i+1].set_ylabel('X₂')
        axes[1, i+1].axis('equal')
        axes[1, i+1].grid(True, alpha=0.3)
    
    # Plot original data again for comparison
    axes[1, 0].scatter(data[:, 0], data[:, 1], alpha=0.6)
    axes[1, 0].set_title('Original Data (Reference)')
    axes[1, 0].set_xlabel('X₁')
    axes[1, 0].set_ylabel('X₂')
    axes[1, 0].axis('equal')
    axes[1, 0].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def whitening_for_gaussian_simplification():
    """
    Demonstrate how whitening simplifies the Gaussian PDF.
    """
    # Original Gaussian with correlation
    mean = np.array([1, 2])
    cov = np.array([[2, 1.2], [1.2, 1.5]])
    
    # Generate data
    np.random.seed(42)
    data = np.random.multivariate_normal(mean, cov, 500)
    
    # Apply whitening
    W, data_mean = compute_whitening_matrix(data, 'ZCA')
    whitened_data = apply_whitening(data, W, data_mean)
    
    # Create evaluation grids
    # Original space
    x1 = np.linspace(data[:, 0].min()-1, data[:, 0].max()+1, 50)
    x2 = np.linspace(data[:, 1].min()-1, data[:, 1].max()+1, 50)
    X1, X2 = np.meshgrid(x1, x2)
    pos_orig = np.dstack((X1, X2))
    
    # Whitened space
    z1 = np.linspace(whitened_data[:, 0].min()-1, whitened_data[:, 0].max()+1, 50)
    z2 = np.linspace(whitened_data[:, 1].min()-1, whitened_data[:, 1].max()+1, 50)
    Z1, Z2 = np.meshgrid(z1, z2)
    pos_white = np.dstack((Z1, Z2))
    
    # Evaluate PDFs
    rv_orig = multivariate_normal(mean, cov)
    pdf_orig = rv_orig.pdf(pos_orig)
    
    # Whitened space has identity covariance and zero mean (approximately)
    rv_white = multivariate_normal([0, 0], np.eye(2))
    pdf_white = rv_white.pdf(pos_white)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original space - data
    axes[0, 0].scatter(data[:, 0], data[:, 1], alpha=0.6)
    axes[0, 0].set_title('Original Data')
    axes[0, 0].set_xlabel('X₁')
    axes[0, 0].set_ylabel('X₂')
    axes[0, 0].axis('equal')
    
    # Original space - PDF contours
    contour1 = axes[0, 1].contour(X1, X2, pdf_orig, levels=10)
    axes[0, 1].scatter(data[:, 0], data[:, 1], alpha=0.4, s=10)
    axes[0, 1].set_title('Original PDF')
    axes[0, 1].set_xlabel('X₁')
    axes[0, 1].set_ylabel('X₂')
    axes[0, 1].axis('equal')
    
    # Original space - 3D PDF
    ax_3d_orig = fig.add_subplot(233, projection='3d')
    surf1 = ax_3d_orig.plot_surface(X1, X2, pdf_orig, cmap='viridis', alpha=0.8)
    ax_3d_orig.set_title('Original PDF (3D)')
    ax_3d_orig.set_xlabel('X₁')
    ax_3d_orig.set_ylabel('X₂')
    ax_3d_orig.set_zlabel('p(x)')
    
    # Whitened space - data
    axes[1, 0].scatter(whitened_data[:, 0], whitened_data[:, 1], alpha=0.6)
    axes[1, 0].set_title('Whitened Data')
    axes[1, 0].set_xlabel('Z₁')
    axes[1, 0].set_ylabel('Z₂')
    axes[1, 0].axis('equal')
    
    # Add unit circles for reference
    circle = plt.Circle((0, 0), 1, fill=False, edgecolor='red', linewidth=2, linestyle='--')
    axes[1, 0].add_patch(circle)
    circle2 = plt.Circle((0, 0), 2, fill=False, edgecolor='red', linewidth=2, linestyle='--')
    axes[1, 0].add_patch(circle2)
    
    # Whitened space - PDF contours
    contour2 = axes[1, 1].contour(Z1, Z2, pdf_white, levels=10)
    axes[1, 1].scatter(whitened_data[:, 0], whitened_data[:, 1], alpha=0.4, s=10)
    axes[1, 1].set_title('Whitened PDF (Identity Covariance)')
    axes[1, 1].set_xlabel('Z₁')
    axes[1, 1].set_ylabel('Z₂')
    axes[1, 1].axis('equal')
    
    # Whitened space - 3D PDF
    ax_3d_white = fig.add_subplot(236, projection='3d')
    surf2 = ax_3d_white.plot_surface(Z1, Z2, pdf_white, cmap='viridis', alpha=0.8)
    ax_3d_white.set_title('Whitened PDF (3D)')
    ax_3d_white.set_xlabel('Z₁')
    ax_3d_white.set_ylabel('Z₂')
    ax_3d_white.set_zlabel('p(z)')
    
    plt.tight_layout()
    plt.show()
    
    # Print the simplified PDF formula
    print("\nGaussian PDF Simplification through Whitening")
    print("=" * 55)
    print("Original PDF:")
    print("p(x) = (1/((2π)^(d/2)|Σ|^(1/2))) * exp(-1/2 * (x-μ)ᵀΣ⁻¹(x-μ))")
    print(f"\nWith μ = {mean} and Σ = \n{cov}")
    print(f"Determinant |Σ| = {np.linalg.det(cov):.3f}")
    
    print("\nAfter whitening (z = Σ^(-1/2)(x-μ)):")
    print("p(z) = (1/(2π)^(d/2)) * exp(-1/2 * zᵀz)")
    print("where z has zero mean and identity covariance")
    
    whitened_cov = np.cov(whitened_data.T)
    print(f"\nWhitened covariance:\n{whitened_cov}")
    print(f"Determinant of whitened covariance: {np.linalg.det(whitened_cov):.6f}")

def high_dimensional_whitening():
    """
    Demonstrate whitening in higher dimensions.
    """
    # Generate high-dimensional correlated data
    np.random.seed(42)
    d = 5  # 5 dimensions
    
    # Create a random covariance matrix
    A = np.random.randn(d, d)
    cov = A @ A.T + 0.1 * np.eye(d)  # Ensure positive definite
    mean = np.random.randn(d)
    
    data = np.random.multivariate_normal(mean, cov, 1000)
    
    # Apply whitening
    W, data_mean = compute_whitening_matrix(data, 'ZCA')
    whitened_data = apply_whitening(data, W, data_mean)
    
    # Analyze results
    orig_cov = np.cov(data.T)
    white_cov = np.cov(whitened_data.T)
    
    print(f"\nHigh-Dimensional Whitening (d={d})")
    print("=" * 40)
    print(f"Original covariance matrix condition number: {np.linalg.cond(orig_cov):.3f}")
    print(f"Whitened covariance matrix condition number: {np.linalg.cond(white_cov):.6f}")
    print(f"Is whitened covariance identity? {np.allclose(white_cov, np.eye(d), atol=1e-6)}")
    
    # Visualize using PCA for 2D projection
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    
    data_2d = pca.fit_transform(data)
    whitened_2d = pca.fit_transform(whitened_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.6)
    axes[0].set_title(f'Original {d}D Data\n(PCA projection to 2D)')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].axis('equal')
    
    axes[1].scatter(whitened_2d[:, 0], whitened_2d[:, 1], alpha=0.6)
    axes[1].set_title(f'Whitened {d}D Data\n(PCA projection to 2D)')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].axis('equal')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function demonstrating whitening transforms."""
    print("Whitening Transform Demonstration")
    print("=" * 40)
    
    # Compare whitening methods
    print("1. Comparing different whitening methods...")
    demonstrate_whitening_methods()
    
    # Gaussian simplification
    print("\n2. Gaussian PDF simplification through whitening...")
    whitening_for_gaussian_simplification()
    
    # High-dimensional example
    print("\n3. High-dimensional whitening...")
    high_dimensional_whitening()

if __name__ == "__main__":
    main()
