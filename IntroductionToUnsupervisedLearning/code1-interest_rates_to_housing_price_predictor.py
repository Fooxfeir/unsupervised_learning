import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

class HousingPricePredictor(nn.Module):
    """
    Neural network to predict housing price index from interest rates
    """
    def __init__(self, hidden_sizes=[128, 64, 32, 16]):
        super(HousingPricePredictor, self).__init__()
        
        layers = []
        input_size = 1  # Interest rate
        
        # Hidden layers with batch normalization
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_size = hidden_size
        
        # Output layer (Housing Price Index)
        layers.append(nn.Linear(input_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def generate_housing_data(n_samples=3000):
    """
    Generate realistic interest rate and housing price data
    Based on empirical relationships observed in real estate markets
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate interest rates (X1) - realistic range and patterns
    # Mix different economic periods
    low_rate_period = np.random.normal(2.5, 0.8, int(0.3 * n_samples))  # Low rate environment
    normal_rate_period = np.random.normal(5.5, 1.2, int(0.5 * n_samples))  # Normal rates
    high_rate_period = np.random.normal(8.5, 1.5, int(0.2 * n_samples))  # High rate period
    
    # Combine and add some extreme events
    interest_rates = np.concatenate([low_rate_period, normal_rate_period, high_rate_period])
    
    # Add occasional rate shocks (financial crises, policy changes)
    shock_indices = np.random.choice(len(interest_rates), int(0.05 * len(interest_rates)), replace=False)
    interest_rates[shock_indices] += np.random.normal(0, 2, len(shock_indices))
    
    # Ensure realistic bounds (0.1% to 15%)
    interest_rates = np.clip(interest_rates, 0.1, 15.0)
    np.random.shuffle(interest_rates)
    
    # Generate Housing Price Index (X2) based on interest rates
    housing_prices = []
    base_price = 100  # Base index value
    
    for rate in interest_rates:
        # Core inverse relationship: higher rates → lower prices
        # Non-linear relationship with different sensitivities at different rate levels
        
        if rate < 3:  # Very low rates - housing boom conditions
            rate_impact = (3 - rate) * 25 + (3 - rate)**2 * 8  # Strong positive impact
            volatility = 8  # High volatility in boom periods
        elif rate < 6:  # Moderate rates - normal market
            rate_impact = (6 - rate) * 12  # Linear relationship
            volatility = 5  # Normal volatility
        elif rate < 10:  # High rates - market cooling
            rate_impact = (6 - rate) * 15 + (rate - 6)**2 * (-2)  # Accelerating negative impact
            volatility = 6  # Increased uncertainty
        else:  # Very high rates - market freeze
            rate_impact = -80 - (rate - 10) * 10  # Severe negative impact
            volatility = 10  # High volatility due to market stress
        
        # Add economic cycle effects (some periods are just better/worse)
        cycle_noise = np.random.normal(0, 3)
        
        # Market sentiment and other factors
        market_noise = np.random.normal(0, volatility)
        
        # Housing price with realistic bounds (30 to 300 index points)
        price = base_price + rate_impact + cycle_noise + market_noise
        price = max(30, min(300, price))  # Realistic bounds
        housing_prices.append(price)
    
    return interest_rates, np.array(housing_prices)

def train_model(model, train_loader, val_loader, epochs=150, lr=0.001):
    """Train the neural network with learning rate scheduling"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                predictions = model(x_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= 20:  # Early stopping
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 25 == 0:
            print(f'Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses

# Generate realistic housing market data
print("Generating realistic interest rate and housing price data...")
interest_rates, housing_prices = generate_housing_data(3000)

print(f"Data Statistics:")
print(f"Interest rates - Mean: {interest_rates.mean():.2f}%, Std: {interest_rates.std():.2f}%")
print(f"Housing prices - Mean: {housing_prices.mean():.1f}, Std: {housing_prices.std():.1f}")
print(f"Correlation: {np.corrcoef(interest_rates, housing_prices)[0,1]:.3f}")

# Prepare data
scaler_x = StandardScaler()
scaler_y = StandardScaler()

# Split data
train_size = int(0.8 * len(interest_rates))
x_train = scaler_x.fit_transform(interest_rates[:train_size].reshape(-1, 1))
y_train = scaler_y.fit_transform(housing_prices[:train_size].reshape(-1, 1))
x_val = scaler_x.transform(interest_rates[train_size:].reshape(-1, 1))
y_val = scaler_y.transform(housing_prices[train_size:].reshape(-1, 1))

# Convert to tensors
x_train_tensor = torch.FloatTensor(x_train)
y_train_tensor = torch.FloatTensor(y_train)
x_val_tensor = torch.FloatTensor(x_val)
y_val_tensor = torch.FloatTensor(y_val)

# Create data loaders
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Initialize and train model
print("\nInitializing housing price prediction model...")
model = HousingPricePredictor(hidden_sizes=[128, 64, 32, 16])
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\nTraining model...")
train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=150)

# Evaluate model
model.eval()
with torch.no_grad():
    # Predictions on validation set
    val_predictions_scaled = model(x_val_tensor)
    val_predictions = scaler_y.inverse_transform(val_predictions_scaled.numpy())
    val_actual = scaler_y.inverse_transform(y_val_tensor.numpy())
    
    # Calculate metrics
    r2 = r2_score(val_actual, val_predictions)
    mae = mean_absolute_error(val_actual, val_predictions)
    rmse = np.sqrt(np.mean((val_actual - val_predictions)**2))
    
    print(f"\nModel Performance on Validation Set:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.1f} index points")
    print(f"Root Mean Square Error: {rmse:.1f} index points")

# Visualization
plt.figure(figsize=(20, 12))

# Plot 1: Training curves
plt.subplot(2, 3, 1)
plt.plot(train_losses, label='Training Loss', alpha=0.8)
plt.plot(val_losses, label='Validation Loss', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss (Scaled)')
plt.title('Training Curves')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted
plt.subplot(2, 3, 2)
plt.scatter(val_actual, val_predictions, alpha=0.6, s=10)
plt.plot([val_actual.min(), val_actual.max()], [val_actual.min(), val_actual.max()], 'r--', lw=2)
plt.xlabel('Actual Housing Price Index')
plt.ylabel('Predicted Housing Price Index')
plt.title(f'Actual vs Predicted Prices (R² = {r2:.3f})')
plt.grid(True, alpha=0.3)

# Plot 3: Prediction function
plt.subplot(2, 3, 3)
rate_range = np.linspace(0.5, 12, 200).reshape(-1, 1)
rate_range_scaled = scaler_x.transform(rate_range)
rate_range_tensor = torch.FloatTensor(rate_range_scaled)

with torch.no_grad():
    price_pred_scaled = model(rate_range_tensor)
    price_pred = scaler_y.inverse_transform(price_pred_scaled.numpy())

plt.plot(rate_range.flatten(), price_pred.flatten(), 'b-', linewidth=2, label='NN Prediction')
plt.scatter(interest_rates[train_size:], housing_prices[train_size:], alpha=0.4, s=8, color='red', label='Validation Data')
plt.xlabel('Interest Rate (%)')
plt.ylabel('Housing Price Index')
plt.title('Learned Relationship: Interest Rate → Housing Price')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Residuals
plt.subplot(2, 3, 4)
residuals = val_actual.flatten() - val_predictions.flatten()
plt.scatter(val_predictions.flatten(), residuals, alpha=0.6, s=10)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Housing Price Index')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

# Plot 5: Distribution of interest rates
plt.subplot(2, 3, 5)
plt.hist(interest_rates, bins=40, alpha=0.7, density=True, label='Interest Rates')
plt.xlabel('Interest Rate (%)')
plt.ylabel('Density')
plt.title('Distribution of Interest Rates')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: Distribution of housing prices
plt.subplot(2, 3, 6)
plt.hist(housing_prices, bins=40, alpha=0.7, density=True, color='orange', label='Housing Price Index')
plt.xlabel('Housing Price Index')
plt.ylabel('Density')
plt.title('Distribution of Housing Prices')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Analysis of key relationships
print("\nKey Economic Insights:")
print("1. Strong inverse relationship: higher interest rates → lower housing prices")
print("2. Non-linear effects: very low rates create housing bubbles")
print("3. Different sensitivity at different rate levels")
print("4. Neural network captures threshold effects and market regimes")

# Test on policy scenarios
policy_scenarios = np.array([[0.5], [2.0], [3.5], [5.0], [7.0], [9.0], [12.0]])
policy_scaled = scaler_x.transform(policy_scenarios)
policy_tensor = torch.FloatTensor(policy_scaled)

with torch.no_grad():
    policy_predictions_scaled = model(policy_tensor)
    policy_predictions = scaler_y.inverse_transform(policy_predictions_scaled.numpy())

print(f"\nHousing Price Predictions for Different Interest Rate Scenarios:")
for i, (rate, price_pred) in enumerate(zip(policy_scenarios.flatten(), policy_predictions.flatten())):
    scenario_desc = ""
    if rate < 1:
        scenario_desc = " (Emergency/Crisis Rates)"
    elif rate < 3:
        scenario_desc = " (Accommodative Policy)"
    elif rate < 6:
        scenario_desc = " (Neutral Policy)"
    elif rate < 8:
        scenario_desc = " (Restrictive Policy)"
    else:
        scenario_desc = " (Very Tight Policy)"
    
    print(f"Interest Rate: {rate:4.1f}%{scenario_desc:25} → Housing Index: {price_pred:.1f}")

print(f"\nPolicy Impact Analysis:")
base_rate = 5.0
base_scaled = scaler_x.transform([[base_rate]])
base_tensor = torch.FloatTensor(base_scaled)

with torch.no_grad():
    base_prediction_scaled = model(base_tensor)
    base_prediction = scaler_y.inverse_transform(base_prediction_scaled.numpy())[0][0]

for rate_change in [-2, -1, +1, +2]:
    new_rate = base_rate + rate_change
    new_scaled = scaler_x.transform([[new_rate]])
    new_tensor = torch.FloatTensor(new_scaled)
    
    with torch.no_grad():
        new_prediction_scaled = model(new_tensor)
        new_prediction = scaler_y.inverse_transform(new_prediction_scaled.numpy())[0][0]
    
    impact = ((new_prediction - base_prediction) / base_prediction) * 100
    print(f"{rate_change:+.1f}% rate change: {impact:+.1f}% housing price impact")
