import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# =====================================================
# 1. Load Dataset
# =====================================================
data = pd.read_csv("md_simulation_data.csv")

# Columns: step temp press KE PE Energy Vol
X = data[["temp", "Vol"]].values
y = data["PE"].values.reshape(-1, 1)

# =====================================================
# 2. Normalize data
# =====================================================
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

# Add bias term (1s column)
X_scaled = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

# =====================================================
# 3. Split into train/val/test
# =====================================================
n_total = len(X_scaled)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val

indices = np.random.permutation(n_total)
train_idx = indices[:n_train]
val_idx = indices[n_train:n_train + n_val]
test_idx = indices[n_train + n_val:]

X_train, y_train = X_scaled[train_idx], y_scaled[train_idx]
X_val, y_val = X_scaled[val_idx], y_scaled[val_idx]
X_test, y_test = X_scaled[test_idx], y_scaled[test_idx]

# =====================================================
# 4. Initialize weights randomly
# =====================================================
np.random.seed(42)
weights = np.random.randn(X_train.shape[1], 1)

# =====================================================
# 5. Training parameters
# =====================================================
lr = 0.001       # learning rate
epochs = 5000    # number of iterations

train_losses = []
val_losses = []
test_losses = []

# =====================================================
# 6. Training loop (manual gradient descent)
# =====================================================
for epoch in range(epochs):
    # --- Forward pass (prediction) ---
    y_pred_train = X_train @ weights

    # --- Compute training loss (MSE) ---
    train_loss = np.mean((y_pred_train - y_train) ** 2)

    # --- Gradient computation ---
    grad = (2 / len(X_train)) * (X_train.T @ (y_pred_train - y_train))

    # --- Update weights manually ---
    weights -= lr * grad

    # --- Validation and test loss ---
    y_pred_val = X_val @ weights
    val_loss = np.mean((y_pred_val - y_val) ** 2)

    y_pred_test = X_test @ weights
    test_loss = np.mean((y_pred_test - y_test) ** 2)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    test_losses.append(test_loss)

    #if epoch % 10 == 0:
        #print(f"Epoch {epoch:03d}: Train={train_loss:.6f}, Val={val_loss:.6f}, Test={test_loss:.6f}")

# =====================================================
# 7. Plot losses per epoch
# =====================================================
plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Linear Regression (Manual Gradient Descent)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("manual_linear_regression_losses.png", dpi=200)
plt.show()

# =====================================================
# 8. Evaluate final model
# =====================================================
y_pred_all_scaled = X_scaled @ weights
y_pred_all = y_scaler.inverse_transform(y_pred_all_scaled)
y_true_all = y

r2 = 1 - np.sum((y_true_all - y_pred_all)**2) / np.sum((y_true_all - np.mean(y_true_all))**2)
rmse = np.sqrt(np.mean((y_true_all - y_pred_all)**2))

print(f"\nFinal R²: {r2:.4f}, RMSE: {rmse:.4f} eV")

# =====================================================
# 9. Plot true vs predicted PE
# =====================================================
plt.figure(figsize=(6,6))
plt.scatter(y_true_all, y_pred_all, alpha=0.7)
plt.plot([y_true_all.min(), y_true_all.max()],
         [y_true_all.min(), y_true_all.max()], 'r--')
plt.xlabel("True PE (eV)")
plt.ylabel("Predicted PE (eV)")
plt.title(f"Manual Linear Regression (R² = {r2:.3f}, RMSE = {rmse:.3f})")
plt.tight_layout()
plt.savefig("manual_linear_regression_pred.png", dpi=200)
plt.show()

# =====================================================
# 10. Print final learned weights
# =====================================================
bias = weights[0][0]
w1 = weights[1][0]
w2 = weights[2][0]
print(f"\nFinal learned equation (scaled units):")
print(f"PE_scaled = {bias:.4f} + ({w1:.4f})*temp_scaled + ({w2:.4f})*Vol_scaled")
