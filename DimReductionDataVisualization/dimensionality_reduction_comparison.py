import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.preprocessing import StandardScaler
import umap
import time

def load_mnist_sample(n_samples=1500):
    """Load a subset of MNIST for visualization"""
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    
    # Sample subset for faster computation
    np.random.seed(42)
    indices = np.random.choice(X.shape[0], n_samples, replace=False)
    X_sample = X[indices] / 255.0  # Normalize
    y_sample = y[indices]
    
    print(f"Using {n_samples} samples from MNIST")
    return X_sample, y_sample

def apply_methods(X):
    """Apply all four dimensionality reduction methods"""
    results = {}
    
    # PCA
    print("Applying PCA...")
    start = time.time()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    results['PCA'] = pca.fit_transform(X_scaled)
    print(f"PCA: {time.time()-start:.1f}s, Variance explained: {pca.explained_variance_ratio_.sum():.3f}")
    
    # MDS
    print("Applying MDS...")
    start = time.time()
    mds = MDS(n_components=2, random_state=42, n_init=1)
    results['MDS'] = mds.fit_transform(X)
    print(f"MDS: {time.time()-start:.1f}s, Stress: {mds.stress_:.0f}")
    
    # t-SNE
    print("Applying t-SNE...")
    start = time.time()
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    results['t-SNE'] = tsne.fit_transform(X)
    print(f"t-SNE: {time.time()-start:.1f}s, KL divergence: {tsne.kl_divergence_:.1f}")
    
    # UMAP
    print("Applying UMAP...")
    start = time.time()
    umap_reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    results['UMAP'] = umap_reducer.fit_transform(X)
    print(f"UMAP: {time.time()-start:.1f}s")
    
    return results

def plot_results(results, y):
    """Create visualization comparing all methods"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    methods = ['PCA', 'MDS', 't-SNE', 'UMAP']
    
    for i, method in enumerate(methods):
        X_reduced = results[method]
        
        # Create scatter plot
        scatter = axes[i].scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                c=y, cmap='tab10', alpha=0.7, s=30)
        axes[i].set_title(f'{method}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Component 1')
        axes[i].set_ylabel('Component 2')
        
        # Remove ticks for cleaner look
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    # Add colorbar
    plt.subplots_adjust(bottom=0.15)  # Make space for colorbar
    cbar = plt.colorbar(scatter, ax=axes, orientation='horizontal', 
                       pad=0.1, shrink=0.8, aspect=30)
    cbar.set_label('Digit Class', fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Ensure colorbar space after tight_layout
    plt.show()

def discuss_methods():
    """Print discussion of each method's characteristics"""
    
    print("\n" + "="*80)
    print("DIMENSIONALITY REDUCTION METHODS: PROS AND CONS")
    print("="*80)
    
    print("\n🔵 PCA (Principal Component Analysis)")
    print("PROS:")
    print("  • Fast and computationally efficient")
    print("  • Interpretable components (linear combinations of features)")
    print("  • Preserves global variance structure")
    print("  • Deterministic results")
    print("  • Good for Gaussian-distributed data")
    
    print("CONS:")
    print("  • Linear method - cannot capture non-linear relationships")
    print("  • May not separate non-linearly separable classes")
    print("  • Assumes linear correlations")
    print("  • Poor for complex manifold structures")
    
    print("\n🟢 MDS (Multidimensional Scaling)")
    print("PROS:")
    print("  • Preserves pairwise distances well")
    print("  • Can handle non-Euclidean distance metrics")
    print("  • Good global structure preservation")
    print("  • Theoretical foundation in distance geometry")
    
    print("CONS:")
    print("  • Computationally expensive O(N³)")
    print("  • Struggles with high-dimensional data")
    print("  • May distort local neighborhoods")
    print("  • Can suffer from crowding problem")
    
    print("\n🔴 t-SNE (t-Distributed Stochastic Neighbor Embedding)")
    print("PROS:")
    print("  • Excellent at preserving local neighborhoods")
    print("  • Creates very clear cluster separations")
    print("  • Handles non-linear manifolds well")
    print("  • Great for visualization and cluster discovery")
    
    print("CONS:")
    print("  • Poor global structure preservation")
    print("  • Distances in embedding space are not meaningful")
    print("  • Sensitive to perplexity parameter")
    print("  • Computationally expensive")
    print("  • Different runs can give different results")
    print("  • Not suitable for downstream ML tasks")
    
    print("\n🟡 UMAP (Uniform Manifold Approximation and Projection)")
    print("PROS:")
    print("  • Preserves both local AND global structure")
    print("  • Faster than t-SNE")
    print("  • More stable and reproducible")
    print("  • Better preserves distances")
    print("  • Good for downstream machine learning")
    print("  • Based on solid topological theory")
    
    print("CONS:")
    print("  • More hyperparameters to tune")
    print("  • Still a relatively new method")
    print("  • Can be sensitive to n_neighbors parameter")
    print("  • May create false connections in sparse data")
    
    print("\n" + "="*80)
    print("SUMMARY RECOMMENDATIONS")
    print("="*80)
    print("🔵 Use PCA when:")
    print("  • You need fast, interpretable results")
    print("  • Your data is approximately linear")
    print("  • You want to understand feature importance")
    
    print("🟢 Use MDS when:")
    print("  • Preserving exact distances is critical")
    print("  • You have a custom distance metric")
    print("  • Dataset is small-medium size")
    
    print("🔴 Use t-SNE when:")
    print("  • You only care about visualization")
    print("  • Finding clusters is the main goal")
    print("  • Local structure is more important than global")
    
    print("🟡 Use UMAP when:")
    print("  • You need both good visualization AND downstream ML")
    print("  • You want to preserve both local and global structure")
    print("  • Speed is important for large datasets")
    print("  • You want more stable, reproducible results")

def main():
    """Main execution function"""
    print("MNIST Dimensionality Reduction Comparison")
    print("=" * 50)
    
    # Load data
    X, y = load_mnist_sample(n_samples=1500)
    
    # Apply all methods
    results = apply_methods(X)
    
    # Visualize results
    print("\nCreating visualizations...")
    plot_results(results, y)
    
    # Discuss methods
    discuss_methods()

if __name__ == "__main__":
    main()