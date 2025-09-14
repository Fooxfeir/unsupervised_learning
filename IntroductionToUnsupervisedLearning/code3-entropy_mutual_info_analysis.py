import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
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

def discretize_data(data, bins=20):
    """Discretize continuous data for entropy calculation"""
    hist, bin_edges = np.histogram(data, bins=bins)
    probabilities = hist / np.sum(hist)
    # Remove zero probabilities to avoid log(0)
    probabilities = probabilities[probabilities > 0]
    return probabilities, bin_edges

def compute_entropy(probabilities):
    """Compute Shannon entropy H(X) = -sum(p * log2(p))"""
    return -np.sum(probabilities * np.log2(probabilities))

def compute_joint_entropy(data1, data2, bins=20):
    """Compute joint entropy H(X,Y)"""
    hist_2d, _, _ = np.histogram2d(data1, data2, bins=bins)
    joint_probs = hist_2d / np.sum(hist_2d)
    joint_probs = joint_probs[joint_probs > 0]
    return -np.sum(joint_probs * np.log2(joint_probs))

def compute_conditional_entropy(data1, data2, bins=20):
    """Compute conditional entropy H(Y|X) = H(X,Y) - H(X)"""
    h_joint = compute_joint_entropy(data1, data2, bins)
    probs1, _ = discretize_data(data1, bins)
    h_marginal = compute_entropy(probs1)
    return h_joint - h_marginal

def compute_mutual_information_discrete(data1, data2, bins=20):
    """Compute mutual information I(X;Y) = H(X) + H(Y) - H(X,Y)"""
    probs1, _ = discretize_data(data1, bins)
    probs2, _ = discretize_data(data2, bins)
    h_x = compute_entropy(probs1)
    h_y = compute_entropy(probs2)
    h_joint = compute_joint_entropy(data1, data2, bins)
    return h_x + h_y - h_joint

# Generate data
print("Generating housing market data...")
interest_rates, housing_prices = generate_housing_data(5000)

# Compute all entropy measures
print("Computing entropy and mutual information measures...")
probs_rates, _ = discretize_data(interest_rates, bins=25)
probs_prices, _ = discretize_data(housing_prices, bins=25)

# Entropies
H_X1 = compute_entropy(probs_rates)
H_X2 = compute_entropy(probs_prices)
H_X1_X2 = compute_joint_entropy(interest_rates, housing_prices, bins=25)
H_X2_given_X1 = compute_conditional_entropy(interest_rates, housing_prices, bins=25)
H_X1_given_X2 = compute_conditional_entropy(housing_prices, interest_rates, bins=25)

# Mutual Information
MI_discrete = compute_mutual_information_discrete(interest_rates, housing_prices, bins=25)
MI_sklearn = mutual_info_regression(interest_rates.reshape(-1, 1), housing_prices)[0]

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Entropy and Mutual Information Analysis', fontsize=16, fontweight='bold')

# Plot 1: Marginal Entropies
ax1 = axes[0, 0]

# Show distributions with entropy values
ax1.hist(interest_rates, bins=30, alpha=0.6, density=True, color='steelblue', 
         label=f'X₁: Interest Rates\nH(X₁) = {H_X1:.2f} bits', edgecolor='black')

# Add secondary y-axis for housing prices
ax1_twin = ax1.twinx()
ax1_twin.hist(housing_prices, bins=30, alpha=0.6, density=True, color='orange', 
              label=f'X₂: Housing Prices\nH(X₂) = {H_X2:.2f} bits', edgecolor='black')

ax1.set_xlabel('Interest Rate (%)', fontsize=12)
ax1.set_ylabel('Density (Interest Rates)', fontsize=12, color='steelblue')
ax1_twin.set_ylabel('Density (Housing Prices)', fontsize=12, color='orange')
ax1.set_title('Marginal Entropies: H(X₁) and H(X₂)', fontsize=14, fontweight='bold')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

ax1.grid(True, alpha=0.3)

# Plot 2: Joint Distribution and Joint Entropy
ax2 = axes[0, 1]

# 2D histogram showing joint distribution
hist_2d, x_edges, y_edges = np.histogram2d(interest_rates, housing_prices, bins=30)
im = ax2.imshow(hist_2d.T, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], 
                cmap='Blues', aspect='auto')
plt.colorbar(im, ax=ax2, label='Count')

ax2.set_xlabel('Interest Rate (%)', fontsize=12)
ax2.set_ylabel('Housing Price Index', fontsize=12)
ax2.set_title(f'Joint Distribution P(X₁,X₂)\nH(X₁,X₂) = {H_X1_X2:.2f} bits', fontsize=14, fontweight='bold')

# Plot 3: Conditional Entropies
ax3 = axes[0, 2]

# Show conditional distributions at different rates
specific_rates = [2.0, 5.0, 8.0]
colors = ['green', 'blue', 'red']
conditional_entropies = []

for i, (rate, color) in enumerate(zip(specific_rates, colors)):
    rate_mask = np.abs(interest_rates - rate) < 0.75
    conditional_prices = housing_prices[rate_mask]
    
    if len(conditional_prices) > 50:
        # Compute conditional entropy for this specific rate
        probs_cond, _ = discretize_data(conditional_prices, bins=15)
        h_cond = compute_entropy(probs_cond)
        conditional_entropies.append(h_cond)
        
        ax3.hist(conditional_prices, bins=20, alpha=0.6, density=True, 
                color=color, label=f'P(X₂|X₁={rate}%)\nH = {h_cond:.2f} bits', 
                edgecolor='black')

ax3.set_xlabel('Housing Price Index', fontsize=12)
ax3.set_ylabel('Density', fontsize=12)
ax3.set_title(f'Conditional Distributions\nAvg H(X₂|X₁) = {H_X2_given_X1:.2f} bits', 
              fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Entropy Relationships (Venn Diagram Style)
ax4 = axes[1, 0]

# Create a conceptual Venn diagram
from matplotlib.patches import Circle
import matplotlib.patches as patches

# Clear the axis for custom drawing
ax4.clear()
ax4.set_xlim(-3, 3)
ax4.set_ylim(-2, 2)

# Draw circles representing entropies
circle1 = Circle((-0.8, 0), 1.2, fill=False, edgecolor='blue', linewidth=3)
circle2 = Circle((0.8, 0), 1.2, fill=False, edgecolor='red', linewidth=3)
ax4.add_patch(circle1)
ax4.add_patch(circle2)

# Add labels
ax4.text(-1.5, 0, f'H(X₁)\n{H_X1:.2f}', ha='center', va='center', fontsize=12, 
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax4.text(1.5, 0, f'H(X₂)\n{H_X2:.2f}', ha='center', va='center', fontsize=12,
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
ax4.text(0, 0, f'I(X₁;X₂)\n{MI_discrete:.2f}', ha='center', va='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Add conditional entropies
ax4.text(-0.4, -0.7, f'H(X₁|X₂)\n{H_X1_given_X2:.2f}', ha='center', va='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
ax4.text(0.4, -0.7, f'H(X₂|X₁)\n{H_X2_given_X1:.2f}', ha='center', va='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.6))

ax4.text(0, 1.5, f'H(X₁,X₂) = {H_X1_X2:.2f} bits', ha='center', va='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

ax4.set_title('Information Theory Relationships', fontsize=14, fontweight='bold')
ax4.set_aspect('equal')
ax4.axis('off')

# Plot 5: Mutual Information Analysis
ax5 = axes[1, 1]

# Show how mutual information changes with binning
bin_sizes = range(10, 51, 5)
mi_values = []

for bins in bin_sizes:
    mi = compute_mutual_information_discrete(interest_rates, housing_prices, bins)
    mi_values.append(mi)

ax5.plot(bin_sizes, mi_values, 'o-', linewidth=2, markersize=8, color='purple')
ax5.axhline(y=MI_sklearn, color='red', linestyle='--', linewidth=2, 
           label=f'Sklearn MI = {MI_sklearn:.3f}')
ax5.set_xlabel('Number of Bins', fontsize=12)
ax5.set_ylabel('Mutual Information (bits)', fontsize=12)
ax5.set_title('Mutual Information vs Discretization', fontsize=14, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Information Reduction
ax6 = axes[1, 2]

# Show how knowing X1 reduces uncertainty in X2
uncertainty_reduction = H_X2 - H_X2_given_X1
percentage_reduction = (uncertainty_reduction / H_X2) * 100

categories = ['H(X₂)', 'H(X₂|X₁)', 'Reduction']
values = [H_X2, H_X2_given_X1, uncertainty_reduction]
colors_bar = ['orange', 'lightcoral', 'green']

bars = ax6.bar(categories, values, color=colors_bar, alpha=0.7, edgecolor='black')
ax6.set_ylabel('Entropy (bits)', fontsize=12)
ax6.set_title(f'Information Gain from Knowing X₁\n{percentage_reduction:.1f}% Uncertainty Reduction', 
              fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3)

# Add value labels on bars
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Comprehensive explanation
print("\n" + "="*80)
print("ENTROPY AND MUTUAL INFORMATION ANALYSIS")
print("="*80)

print(f"""
📊 ENTROPY MEASURES (in bits):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• H(X₁) = {H_X1:.3f} bits    - Uncertainty in interest rates
• H(X₂) = {H_X2:.3f} bits    - Uncertainty in housing prices  
• H(X₁,X₂) = {H_X1_X2:.3f} bits - Joint uncertainty
• H(X₂|X₁) = {H_X2_given_X1:.3f} bits - Uncertainty in prices GIVEN rates
• H(X₁|X₂) = {H_X1_given_X2:.3f} bits - Uncertainty in rates GIVEN prices
""")

print(f"""
🔗 MUTUAL INFORMATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• I(X₁;X₂) = {MI_discrete:.3f} bits (discrete approximation)
• I(X₁;X₂) = {MI_sklearn:.3f} bits (sklearn estimate)

Mutual Information measures how much information X₁ and X₂ share.
It's the reduction in uncertainty about one variable when we observe the other.
""")

print(f"""
🎯 KEY RELATIONSHIPS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. I(X₁;X₂) = H(X₁) + H(X₂) - H(X₁,X₂) = {H_X1:.3f} + {H_X2:.3f} - {H_X1_X2:.3f} = {MI_discrete:.3f}
2. I(X₁;X₂) = H(X₂) - H(X₂|X₁) = {H_X2:.3f} - {H_X2_given_X1:.3f} = {H_X2 - H_X2_given_X1:.3f}
3. H(X₁,X₂) = H(X₁) + H(X₂|X₁) = {H_X1:.3f} + {H_X2_given_X1:.3f} = {H_X1 + H_X2_given_X1:.3f}
""")

uncertainty_reduction = H_X2 - H_X2_given_X1
percentage_reduction = (uncertainty_reduction / H_X2) * 100

print(f"""
💡 PRACTICAL INTERPRETATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 ENTROPY INTERPRETATION:
• Higher entropy = More uncertainty/randomness
• H(X₂) = {H_X2:.2f}: Housing prices have this much uncertainty
• H(X₂|X₁) = {H_X2_given_X1:.2f}: After knowing interest rates, uncertainty drops

📉 UNCERTAINTY REDUCTION:
• Knowing interest rates reduces housing price uncertainty by {uncertainty_reduction:.2f} bits
• This is a {percentage_reduction:.1f}% reduction in uncertainty!
• This validates why interest rates are good predictors of housing prices

🤖 FOR NEURAL NETWORKS:
• High mutual information → X₁ is informative for predicting X₂
• Low conditional entropy H(X₂|X₁) → Good predictive relationship
• This explains why our neural network can learn a meaningful X₁ → X₂ mapping
""")

print(f"""
🔢 COMPARISON WITH INDEPENDENCE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If X₁ and X₂ were independent:
• I(X₁;X₂) would be ≈ 0
• H(X₂|X₁) would equal H(X₂)
• H(X₁,X₂) would equal H(X₁) + H(X₂)

Our data shows:
• I(X₁;X₂) = {MI_discrete:.3f} >> 0  ✓ Strong dependence
• H(X₂|X₁) = {H_X2_given_X1:.3f} < H(X₂) = {H_X2:.3f}  ✓ X₁ reduces uncertainty
• H(X₁,X₂) = {H_X1_X2:.3f} < H(X₁) + H(X₂) = {H_X1 + H_X2:.3f}  ✓ Variables share information

This confirms that interest rates and housing prices are NOT independent!
""")

print(f"""
🎨 PLOT EXPLANATIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. MARGINAL ENTROPIES: Shows individual uncertainty in each variable
2. JOINT DISTRIBUTION: Visualizes P(X₁,X₂) and joint entropy
3. CONDITIONAL DISTRIBUTIONS: Shows how H(X₂|X₁) varies with X₁
4. VENN DIAGRAM: Visual representation of entropy relationships
5. MI vs BINNING: Shows how discretization affects MI estimation
6. UNCERTAINTY REDUCTION: Quantifies information gain from knowing X₁
""")
