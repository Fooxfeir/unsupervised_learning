import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set clean style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def generate_housing_data(n_samples=5000):
    """Generate the same housing data as before"""
    np.random.seed(42)
    
    # Generate interest rates
    low_rate_period = np.random.normal(2.5, 0.8, int(0.3 * n_samples))
    normal_rate_period = np.random.normal(5.5, 1.2, int(0.5 * n_samples))
    high_rate_period = np.random.normal(8.5, 1.5, int(0.2 * n_samples))
    
    interest_rates = np.concatenate([low_rate_period, normal_rate_period, high_rate_period])
    shock_indices = np.random.choice(len(interest_rates), int(0.05 * len(interest_rates)), replace=False)
    interest_rates[shock_indices] += np.random.normal(0, 2, len(shock_indices))
    interest_rates = np.clip(interest_rates, 0.1, 15.0)
    np.random.shuffle(interest_rates)
    
    # Generate housing prices
    housing_prices = []
    base_price = 100
    
    for rate in interest_rates:
        if rate < 3:
            rate_impact = (3 - rate) * 25 + (3 - rate)**2 * 8
            volatility = 8
        elif rate < 6:
            rate_impact = (6 - rate) * 12
            volatility = 5
        elif rate < 10:
            rate_impact = (6 - rate) * 15 + (rate - 6)**2 * (-2)
            volatility = 6
        else:
            rate_impact = -80 - (rate - 10) * 10
            volatility = 10
        
        cycle_noise = np.random.normal(0, 3)
        market_noise = np.random.normal(0, volatility)
        price = base_price + rate_impact + cycle_noise + market_noise
        price = max(30, min(300, price))
        housing_prices.append(price)
    
    return interest_rates, np.array(housing_prices)

def fit_gaussian(data):
    """Fit a Gaussian distribution to data"""
    mu = np.mean(data)
    sigma = np.std(data)
    return mu, sigma

def gaussian_pdf(x, mu, sigma):
    """Gaussian probability density function"""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def kl_divergence_gaussian(mu1, sigma1, mu2, sigma2):
    """
    Analytical KL divergence between two Gaussian distributions
    KL(N(mu1, sigma1^2) || N(mu2, sigma2^2))
    """
    return np.log(sigma2/sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5

def empirical_kl_divergence(data1, data2, bins=50):
    """
    Estimate KL divergence between two empirical distributions
    KL(P || Q) = sum(P * log(P/Q))
    """
    # Create common bins
    all_data = np.concatenate([data1, data2])
    bin_edges = np.linspace(all_data.min(), all_data.max(), bins+1)
    
    # Get empirical probabilities
    p, _ = np.histogram(data1, bins=bin_edges, density=True)
    q, _ = np.histogram(data2, bins=bin_edges, density=True)
    
    # Normalize to sum to 1
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p = p + eps
    q = q + eps
    
    # Compute KL divergence
    kl = np.sum(p * np.log(p / q))
    return kl

# Generate data
print("Generating housing market data...")
interest_rates, housing_prices = generate_housing_data(5000)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('KL Divergence Analysis: Comparing Probability Distributions', fontsize=16, fontweight='bold')

# Plot 1: Marginal distributions P(X1) vs P(X2)
ax1 = axes[0, 0]

# Fit Gaussians to both marginal distributions
mu1, sigma1 = fit_gaussian(interest_rates)
mu2, sigma2 = fit_gaussian(housing_prices)

# Create normalized versions for fair comparison
rates_norm = (interest_rates - interest_rates.mean()) / interest_rates.std()
prices_norm = (housing_prices - housing_prices.mean()) / housing_prices.std()

# Plot normalized distributions
ax1.hist(rates_norm, bins=40, alpha=0.6, density=True, color='steelblue', 
         label='Normalized P(X₁)', edgecolor='black')
ax1.hist(prices_norm, bins=40, alpha=0.6, density=True, color='orange', 
         label='Normalized P(X₂)', edgecolor='black')

# Add fitted Gaussian curves
x_range = np.linspace(-4, 4, 200)
ax1.plot(x_range, gaussian_pdf(x_range, 0, 1), 'b-', linewidth=2, label='Fitted N(0,1) for X₁')
ax1.plot(x_range, gaussian_pdf(x_range, 0, 1), 'r-', linewidth=2, label='Fitted N(0,1) for X₂')

ax1.set_xlabel('Normalized Value', fontsize=12)
ax1.set_ylabel('Probability Density', fontsize=12)
ax1.set_title('Marginal Distributions: P(X₁) vs P(X₂)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Compute KL divergence between marginals (using normalized data)
kl_marginals = empirical_kl_divergence(rates_norm, prices_norm)
ax1.text(0.02, 0.98, f'KL(P(X₁)||P(X₂)) ≈ {kl_marginals:.3f}', 
         transform=ax1.transAxes, verticalalignment='top', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# Plot 2: Conditional distributions P(X2|X1) at different rates
ax2 = axes[0, 1]

specific_rates = [2.0, 5.0, 8.0]
colors = ['green', 'blue', 'red']
conditional_data = {}

for rate, color in zip(specific_rates, colors):
    # Get conditional data
    rate_mask = np.abs(interest_rates - rate) < 0.75
    conditional_prices = housing_prices[rate_mask]
    conditional_data[rate] = conditional_prices
    
    if len(conditional_prices) > 20:
        ax2.hist(conditional_prices, bins=20, alpha=0.6, density=True, 
                color=color, label=f'P(X₂|X₁={rate}%)', edgecolor='black')

ax2.set_xlabel('Housing Price Index', fontsize=12)
ax2.set_ylabel('Probability Density', fontsize=12)
ax2.set_title('Conditional Distributions P(X₂|X₁) at Different Rates', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: KL divergences between conditional distributions
ax3 = axes[1, 0]

# Compute pairwise KL divergences between conditional distributions
kl_matrix = np.zeros((len(specific_rates), len(specific_rates)))
rate_pairs = []
kl_values = []

for i, rate1 in enumerate(specific_rates):
    for j, rate2 in enumerate(specific_rates):
        if i != j and rate1 in conditional_data and rate2 in conditional_data:
            data1 = conditional_data[rate1]
            data2 = conditional_data[rate2]
            
            if len(data1) > 20 and len(data2) > 20:
                kl_val = empirical_kl_divergence(data1, data2)
                kl_matrix[i, j] = kl_val
                
                if i < j:  # Only store upper triangle to avoid duplicates
                    rate_pairs.append(f'{rate1}% → {rate2}%')
                    kl_values.append(kl_val)

# Create bar plot of KL divergences
if rate_pairs:
    bars = ax3.bar(range(len(rate_pairs)), kl_values, 
                   color=['lightcoral', 'lightblue', 'lightgreen'])
    ax3.set_xticks(range(len(rate_pairs)))
    ax3.set_xticklabels(rate_pairs, rotation=45)
    ax3.set_ylabel('KL Divergence', fontsize=12)
    ax3.set_title('KL(P(X₂|X₁=a) || P(X₂|X₁=b))', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, kl_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Visual representation of KL divergence concept
ax4 = axes[1, 1]

# Show two specific conditional distributions for KL visualization
rate_a, rate_b = 2.0, 8.0
data_a = conditional_data[rate_a]
data_b = conditional_data[rate_b]

# Create smooth probability curves
x_smooth = np.linspace(40, 200, 200)
mu_a, sigma_a = fit_gaussian(data_a)
mu_b, sigma_b = fit_gaussian(data_b)

p_a = gaussian_pdf(x_smooth, mu_a, sigma_a)
p_b = gaussian_pdf(x_smooth, mu_b, sigma_b)

ax4.plot(x_smooth, p_a, 'g-', linewidth=3, label=f'P(X₂|X₁={rate_a}%) - Distribution A')
ax4.plot(x_smooth, p_b, 'r-', linewidth=3, label=f'P(X₂|X₁={rate_b}%) - Distribution B')

# Fill areas to show difference
ax4.fill_between(x_smooth, 0, p_a, alpha=0.3, color='green')
ax4.fill_between(x_smooth, 0, p_b, alpha=0.3, color='red')

ax4.set_xlabel('Housing Price Index', fontsize=12)
ax4.set_ylabel('Probability Density', fontsize=12)
ax4.set_title('KL Divergence Visualization: How Different Are These Distributions?', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Compute and display KL divergence
kl_ab = kl_divergence_gaussian(mu_a, sigma_a, mu_b, sigma_b)
kl_ba = kl_divergence_gaussian(mu_b, sigma_b, mu_a, sigma_a)

ax4.text(0.02, 0.98, f'KL(A||B) = {kl_ab:.3f}\nKL(B||A) = {kl_ba:.3f}\n(Asymmetric!)', 
         transform=ax4.transAxes, verticalalignment='top', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

plt.tight_layout()
plt.show()

# Detailed explanation
print("\n" + "="*80)
print("KL DIVERGENCE ANALYSIS EXPLANATION")
print("="*80)

print("""
🔍 IMPORTANT CLARIFICATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KL divergence is defined between PROBABILITY DISTRIBUTIONS, not random variables.
We cannot compute "KL between X₁ and X₂" directly, but we can compute:

1. KL(P(X₁) || P(X₂)) - Between their marginal distributions
2. KL(P(X₂|X₁=a) || P(X₂|X₁=b)) - Between conditional distributions
""")

print(f"""
📊 PLOT 1 - Marginal Distribution Comparison: KL(P(X₁) || P(X₂))
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This compares the overall distributions of interest rates vs housing prices.
Since they have different units and scales, we normalize both to N(0,1).

Result: KL ≈ {kl_marginals:.3f}
Interpretation: This measures how "different" the shapes of the two marginal 
distributions are, but it's not very meaningful since X₁ and X₂ represent 
completely different quantities.
""")

print("""
🏠 PLOT 2 - Conditional Distributions P(X₂|X₁) 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This shows how the distribution of housing prices changes for different 
interest rates. Notice how:
• Green (2% rates): Distribution centered around ~160 (high prices)
• Blue (5% rates): Distribution centered around ~100 (moderate prices)  
• Red (8% rates): Distribution centered around ~70 (low prices)

This is much more meaningful than comparing marginal distributions!
""")

if rate_pairs:
    print(f"""
📏 PLOT 3 - KL Divergences Between Conditional Distributions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This shows how "different" the conditional distributions are from each other:
""")
    for pair, kl_val in zip(rate_pairs, kl_values):
        print(f"• KL({pair}) = {kl_val:.3f}")
    
    print(f"""
Higher KL values mean the distributions are more different.
The largest difference is between extreme rate scenarios (2% vs 8%).
""")

print(f"""
🎯 PLOT 4 - KL Divergence Visualization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This visualizes the concept of KL divergence between two specific distributions.

Key insights:
• KL(A||B) = {kl_ab:.3f} ≠ KL(B||A) = {kl_ba:.3f}
• KL divergence is ASYMMETRIC! 
• It measures how much information is lost when we approximate distribution A with B
• The filled areas show how different the probability mass is distributed

💡 Mathematical Interpretation:
KL(P||Q) = ∫ P(x) log(P(x)/Q(x)) dx
• When P(x) is high but Q(x) is low → large positive contribution
• When P(x) is low → small contribution regardless of Q(x)
• Always ≥ 0, equals 0 only when P = Q
""")

print("""
🔑 KEY TAKEAWAYS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. KL divergence compares probability distributions, not random variables
2. Most meaningful application here: comparing P(X₂|X₁=a) vs P(X₂|X₁=b)
3. This quantifies how much the housing price distribution changes with rates
4. Large KL values indicate the interest rate strongly affects housing prices
5. Asymmetry of KL shows direction matters in distribution comparison
""")
