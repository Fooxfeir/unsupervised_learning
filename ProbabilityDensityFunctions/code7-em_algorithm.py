"""
Expectation-Maximization Algorithm for Mixture of Gaussians
===========================================================

This module implements the EM algorithm from scratch for fitting
Mixture of Gaussians models.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
import seaborn as sns

class GaussianMixtureEM:
    """
    Gaussian Mixture Model fitted using Expectation-Maximization algorithm.
    """
    
    def __init__(self, n_components=2, max_iter=100, tol=1e-6, random_state=None):
        """
        Initialize GMM parameters.
        
        Parameters:
        -----------
        n_components : int
            Number of Gaussian components
        max_iter : int
            Maximum number of EM iterations
        tol : float
            Convergence tolerance
        random_state : int
            Random seed for reproducibility
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # Will be set during fitting
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.converged_ = False
        self.n_iter_ = 0
        self.log_likelihood_history_ = []
        
    def _initialize_parameters(self, X):
        """
        Initialize parameters for EM algorithm.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        
        # Initialize weights uniformly
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        # Initialize means by randomly selecting data points
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[indices].copy()
        
        # Initialize covariances as identity matrices scaled by data variance
        data_var = np.var(X, axis=0).mean()
        self.covariances_ = np.array([data_var * np.eye(n_features) 
                                    for _ in range(self.n_components)])
    
    def _e_step(self, X):
        """
        Expectation step: compute posterior probabilities.
        """
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        # Compute weighted likelihoods for each component
        weighted_likelihoods = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            try:
                # Compute likelihood for component k
                rv = multivariate_normal(self.means_[k], self.covariances_[k])
                weighted_likelihoods[:, k] = self.weights_[k] * rv.pdf(X)
            except np.linalg.LinAlgError:
                # Handle singular covariance matrix
                weighted_likelihoods[:, k] = 1e-10
        
        # Compute responsibilities (posterior probabilities)
        total_likelihood = np.sum(weighted_likelihoods, axis=1, keepdims=True)
        total_likelihood = np.maximum(total_likelihood, 1e-10)  # Avoid division by zero
        
        responsibilities = weighted_likelihoods / total_likelihood
        
        return responsibilities
    
    def _m_step(self, X, responsibilities):
        """
        Maximization step: update parameters.
        """
        n_samples, n_features = X.shape
        
        # Effective number of points assigned to each component
        N_k = np.sum(responsibilities, axis=0)
        
        # Update weights
        self.weights_ = N_k / n_samples
        
        # Update means
        self.means_ = np.zeros((self.n_components, n_features))
        for k in range(self.n_components):
            if N_k[k] > 1e-10:
                self.means_[k] = np.sum(responsibilities[:, k:k+1] * X, axis=0) / N_k[k]
        
        # Update covariances
        self.covariances_ = np.zeros((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            if N_k[k] > 1e-10:
                diff = X - self.means_[k]
                # Weighted covariance
                self.covariances_[k] = np.dot(responsibilities[:, k] * diff.T, diff) / N_k[k]
                
                # Add regularization to prevent singular matrices
                self.covariances_[k] += 1e-6 * np.eye(n_features)
            else:
                # If component has no points, reset to identity
                self.covariances_[k] = np.eye(n_features)
    
    def _compute_log_likelihood(self, X):
        """
        Compute log-likelihood of data under current model.
        """
        n_samples = X.shape[0]
        log_likelihood = 0
        
        for i in range(n_samples):
            likelihood_i = 0
            for k in range(self.n_components):
                try:
                    rv = multivariate_normal(self.means_[k], self.covariances_[k])
                    likelihood_i += self.weights_[k] * rv.pdf(X[i])
                except:
                    likelihood_i += 1e-10
            
            log_likelihood += np.log(max(likelihood_i, 1e-10))
        
        return log_likelihood
    
    def fit(self, X):
        """
        Fit Gaussian Mixture Model using EM algorithm.
        """
        # Initialize parameters
        self._initialize_parameters(X)
        
        # Initial log-likelihood
        prev_log_likelihood = self._compute_log_likelihood(X)
        self.log_likelihood_history_ = [prev_log_likelihood]
        
        print(f"Initial log-likelihood: {prev_log_likelihood:.4f}")
        
        # EM iterations
        for iteration in range(self.max_iter):
            # E-step
            responsibilities = self._e_step(X)
            
            # M-step
            self._m_step(X, responsibilities)
            
            # Compute log-likelihood
            current_log_likelihood = self._compute_log_likelihood(X)
            self.log_likelihood_history_.append(current_log_likelihood)
            
            # Check convergence
            improvement = current_log_likelihood - prev_log_likelihood
            print(f"Iteration {iteration + 1}: log-likelihood = {current_log_likelihood:.4f}, "
                  f"improvement = {improvement:.6f}")
            
            if abs(improvement) < self.tol:
                self.converged_ = True
                break
                
            prev_log_likelihood = current_log_likelihood
        
        self.n_iter_ = iteration + 1
        
        if self.converged_:
            print(f"EM converged after {self.n_iter_} iterations")
        else:
            print(f"EM did not converge after {self.max_iter} iterations")
    
    def predict_proba(self, X):
        """
        Predict posterior probabilities for new data.
        """
        return self._e_step(X)
    
    def score_samples(self, X):
        """
        Compute log-likelihood of each sample.
        """
        n_samples = X.shape[0]
        log_likelihoods = np.zeros(n_samples)
        
        for i in range(n_samples):
            likelihood_i = 0
            for k in range(self.n_components):
                try:
                    rv = multivariate_normal(self.means_[k], self.covariances_[k])
                    likelihood_i += self.weights_[k] * rv.pdf(X[i])
                except:
                    likelihood_i += 1e-10
            
            log_likelihoods[i] = np.log(max(likelihood_i, 1e-10))
        
        return log_likelihoods

def generate_mixture_data():
    """Generate synthetic data from a mixture of Gaussians."""
    np.random.seed(42)
    
    # Component 1: Regular customers
    n1 = 200
    mean1 = np.array([50, 10])
    cov1 = np.array([[100, 20], [20, 25]])
    cluster1 = np.random.multivariate_normal(mean1, cov1, n1)
    
    # Component 2: Premium customers  
    n2 = 150
    mean2 = np.array([120, 25])
    cov2 = np.array([[200, -30], [-30, 40]])
    cluster2 = np.random.multivariate_normal(mean2, cov2, n2)
    
    # Component 3: Occasional customers
    n3 = 100
    mean3 = np.array([30, 5])
    cov3 = np.array([[50, 10], [10, 15]])
    cluster3 = np.random.multivariate_normal(mean3, cov3, n3)
    
    # Combine data
    data = np.vstack([cluster1, cluster2, cluster3])
    true_labels = np.hstack([np.zeros(n1), np.ones(n2), np.full(n3, 2)])
    
    return data, true_labels

def compare_with_sklearn(data):
    """Compare our implementation with sklearn."""
    print("Comparing with sklearn GaussianMixture...")
    
    # Our implementation
    our_gmm = GaussianMixtureEM(n_components=3, random_state=42)
    our_gmm.fit(data)
    
    # Sklearn implementation
    sklearn_gmm = GaussianMixture(n_components=3, random_state=42, max_iter=100)
    sklearn_gmm.fit(data)
    
    # Compare log-likelihoods
    our_ll = our_gmm.score_samples(data).mean()
    sklearn_ll = sklearn_gmm.score_samples(data).mean()
    
    print(f"Our implementation mean log-likelihood: {our_ll:.4f}")
    print(f"Sklearn mean log-likelihood: {sklearn_ll:.4f}")
    print(f"Difference: {abs(our_ll - sklearn_ll):.6f}")
    
    return our_gmm, sklearn_gmm

def visualize_em_evolution(data, true_labels):
    """Visualize how EM algorithm evolves over iterations."""
    print("Visualizing EM evolution...")
    
    # Custom EM with tracking
    class TrackingGMM(GaussianMixtureEM):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.parameter_history_ = []
        
        def _m_step(self, X, responsibilities):
            super()._m_step(X, responsibilities)
            # Store current parameters
            self.parameter_history_.append({
                'weights': self.weights_.copy(),
                'means': self.means_.copy(),
                'covariances': self.covariances_.copy()
            })
    
    # Fit tracking GMM
    tracking_gmm = TrackingGMM(n_components=3, max_iter=20, random_state=42)
    tracking_gmm.fit(data)
    
    # Plot evolution
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    iterations_to_show = [0, 2, 5, 10, 15, -1]  # -1 for final
    
    for idx, iter_idx in enumerate(iterations_to_show):
        row = idx // 3
        col = idx % 3
        
        if iter_idx == -1:
            iter_idx = len(tracking_gmm.parameter_history_) - 1
            title = f"Final (iter {iter_idx + 1})"
        else:
            title = f"Iteration {iter_idx + 1}"
        
        if iter_idx < len(tracking_gmm.parameter_history_):
            params = tracking_gmm.parameter_history_[iter_idx]
            
            # Plot data
            scatter = axes[row, col].scatter(data[:, 0], data[:, 1], 
                                           c=true_labels, alpha=0.6, s=20)
            
            # Plot component means
            for k in range(3):
                axes[row, col].scatter(params['means'][k, 0], params['means'][k, 1],
                                     marker='x', s=200, linewidth=3, 
                                     color=f'C{k}', label=f'Component {k+1}')
            
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel('Spending ($)')
            axes[row, col].set_ylabel('Frequency')
            if idx == 0:
                axes[row, col].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot log-likelihood evolution
    plt.figure(figsize=(10, 6))
    plt.plot(tracking_gmm.log_likelihood_history_, 'o-')
    plt.xlabel('Iteration')
    plt.ylabel('Log-likelihood')
    plt.title('EM Algorithm Convergence')
    plt.grid(True)
    plt.show()
    
    return tracking_gmm

def comprehensive_analysis(data, true_labels):
    """Perform comprehensive analysis of the fitted model."""
    print("Performing comprehensive analysis...")
    
    # Fit final model
    gmm = GaussianMixtureEM(n_components=3, random_state=42)
    gmm.fit(data)
    
    # Create grid for contour plots
    x_min, x_max = data[:, 0].min() - 20, data[:, 0].max() + 20
    y_min, y_max = data[:, 1].min() - 10, data[:, 1].max() + 10
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Evaluate PDF on grid
    log_probs = gmm.score_samples(grid_points)
    probs = np.exp(log_probs).reshape(xx.shape)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Original data with true clusters
    scatter = axes[0, 0].scatter(data[:, 0], data[:, 1], c=true_labels, alpha=0.7)
    axes[0, 0].set_title('True Customer Segments')
    axes[0, 0].set_xlabel('Monthly Spending ($)')
    axes[0, 0].set_ylabel('Visit Frequency')
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # 2. Fitted model contours
    contour = axes[0, 1].contour(xx, yy, probs, levels=10)
    axes[0, 1].scatter(data[:, 0], data[:, 1], alpha=0.4, s=15)
    # Plot component means
    for k in range(3):
        axes[0, 1].scatter(gmm.means_[k, 0], gmm.means_[k, 1],
                          marker='x', s=200, linewidth=3, color='red')
    axes[0, 1].set_title('Fitted GMM Contours')
    axes[0, 1].set_xlabel('Monthly Spending ($)')
    axes[0, 1].set_ylabel('Visit Frequency')
    
    # 3. 3D surface
    ax_3d = fig.add_subplot(233, projection='3d')
    surface = ax_3d.plot_surface(xx, yy, probs, cmap='viridis', alpha=0.8)
    ax_3d.set_title('PDF as 3D Surface')
    ax_3d.set_xlabel('Spending ($)')
    ax_3d.set_ylabel('Frequency')
    ax_3d.set_zlabel('Density')
    
    # 4. Component responsibilities
    responsibilities = gmm.predict_proba(data)
    predicted_labels = np.argmax(responsibilities, axis=1)
    scatter2 = axes[1, 0].scatter(data[:, 0], data[:, 1], c=predicted_labels, alpha=0.7)
    axes[1, 0].set_title('Predicted Clusters')
    axes[1, 0].set_xlabel('Monthly Spending ($)')
    axes[1, 0].set_ylabel('Visit Frequency')
    plt.colorbar(scatter2, ax=axes[1, 0])
    
    # 5. Individual component PDFs
    axes[1, 1].contour(xx, yy, probs, levels=10, alpha=0.5, colors='gray')
    for k in range(3):
        # Individual component
        component_probs = np.zeros(grid_points.shape[0])
        for i, point in enumerate(grid_points):
            rv = multivariate_normal(gmm.means_[k], gmm.covariances_[k])
            component_probs[i] = gmm.weights_[k] * rv.pdf(point)
        component_probs = component_probs.reshape(xx.shape)
        axes[1, 1].contour(xx, yy, component_probs, levels=5, 
                          colors=[f'C{k}'], alpha=0.7)
    axes[1, 1].scatter(data[:, 0], data[:, 1], alpha=0.3, s=10)
    axes[1, 1].set_title('Individual Components')
    axes[1, 1].set_xlabel('Monthly Spending ($)')
    axes[1, 1].set_ylabel('Visit Frequency')
    
    # 6. Model parameters summary
    axes[1, 2].axis('off')
    param_text = "Model Parameters:\n\n"
    for k in range(3):
        eigenvals = np.linalg.eigvals(gmm.covariances_[k])
        param_text += f"Component {k+1}:\n"
        param_text += f"  Weight: {gmm.weights_[k]:.3f}\n"
        param_text += f"  Mean: [{gmm.means_[k, 0]:.1f}, {gmm.means_[k, 1]:.1f}]\n"
        param_text += f"  Eigenvalues: [{eigenvals[0]:.2f}, {eigenvals[1]:.2f}]\n\n"
    
    param_text += f"Total Log-likelihood: {gmm.log_likelihood_history_[-1]:.2f}\n"
    param_text += f"Converged: {gmm.converged_}\n"
    param_text += f"Iterations: {gmm.n_iter_}"
    
    axes[1, 2].text(0.1, 0.9, param_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis
    print(f"\nDetailed Model Analysis:")
    print(f"Final log-likelihood: {gmm.log_likelihood_history_[-1]:.4f}")
    print(f"Converged: {gmm.converged_} in {gmm.n_iter_} iterations")
    
    for k in range(3):
        eigenvals = np.linalg.eigvals(gmm.covariances_[k])
        print(f"\nComponent {k+1}:")
        print(f"  Weight: {gmm.weights_[k]:.3f}")
        print(f"  Mean: {gmm.means_[k]}")
        print(f"  Covariance eigenvalues: [{eigenvals[0]:.3f}, {eigenvals[1]:.3f}]")
        print(f"  Covariance condition number: {np.linalg.cond(gmm.covariances_[k]):.2f}")
    
    # Compute accuracy if true labels available
    from sklearn.metrics import adjusted_rand_score
    predicted_labels = np.argmax(responsibilities, axis=1)
    ari = adjusted_rand_score(true_labels, predicted_labels)
    print(f"\nAdjusted Rand Index: {ari:.3f}")
    
    return gmm

def main():
    """Main function demonstrating EM algorithm."""
    print("Expectation-Maximization Algorithm for Gaussian Mixtures")
    print("=" * 60)
    
    # Generate data
    data, true_labels = generate_mixture_data()
    print(f"Generated {len(data)} customer data points")
    
    # Basic EM demonstration
    print("\n1. Basic EM Algorithm Implementation:")
    basic_gmm = GaussianMixtureEM(n_components=3, random_state=42)
    basic_gmm.fit(data)
    
    # Compare with sklearn
    print("\n2. Comparison with sklearn:")
    our_gmm, sklearn_gmm = compare_with_sklearn(data)
    
    # Visualize evolution
    print("\n3. EM Algorithm Evolution:")
    tracking_gmm = visualize_em_evolution(data, true_labels)
    
    # Comprehensive analysis
    print("\n4. Comprehensive Analysis:")
    final_gmm = comprehensive_analysis(data, true_labels)

if __name__ == "__main__":
    main()
