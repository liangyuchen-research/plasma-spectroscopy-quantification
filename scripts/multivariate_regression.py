"""Preserved research analysis; see docs/reproduction.md for assumptions."""

from pathlib import Path
import sys

REPOSITORY_ROOT = next(
    candidate
    for candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents)
    if (candidate / "research_paths.py").is_file()
)
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from research_paths import data_file, output_file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the regression dataset
new_data_path = data_file("regression/multivariate_regression.csv")
new_data = pd.read_csv(new_data_path)

# Split training concentrations (0, 5, 10, 20) and testing concentrations (7.5, 15)
new_original_data = new_data.loc[new_data["Cu"].isin([0, 5, 10, 20])].copy()
new_test_data = new_data.loc[new_data["Cu"].isin([7.5, 15])].copy()

# Fit the regression model to the training observations
X_train_new = new_original_data[["Conductivity", "Cu-327nm_predict"]]
y_train_new = new_original_data["Cu"]

regressor_new = LinearRegression()
regressor_new.fit(X_train_new, y_train_new)

# Predict the training and testing observations
new_original_data["Predicted_Concentration"] = regressor_new.predict(X_train_new)
new_test_data["Predicted_Concentration"] = regressor_new.predict(
    new_test_data[["Conductivity", "Cu-327nm_predict"]]
)


# Calculate means and standard deviations for groups of 30 observations
def compute_avg_std(data, columns):
    grouped = data.groupby(np.arange(len(data)) // 30)
    avg = grouped[columns].mean()
    std = grouped[columns].std()
    avg["Cu"] = grouped["Cu"].mean()
    avg["Conductivity"] = grouped["Conductivity"].mean()
    avg["Cu-327nm_predict"] = grouped["Cu-327nm_predict"].mean()
    return avg, std


def compute_avg_std_test(data, columns):
    grouped = data.groupby(np.arange(len(data)) // 10)
    avg = grouped[columns].mean()
    std = grouped[columns].std()
    avg["Cu"] = grouped["Cu"].mean()
    avg["Conductivity"] = grouped["Conductivity"].mean()
    avg["Cu-327nm_predict"] = grouped["Cu-327nm_predict"].mean()
    return avg, std


avg_original_data, std_original_data = compute_avg_std(
    new_original_data, ["Predicted_Concentration"]
)
avg_test_data, std_test_data = compute_avg_std_test(new_test_data, ["Predicted_Concentration"])

# Display the group means and standard deviations
print("Averaged Original Data (Predicted Concentration):")
print(avg_original_data)
print("Standard Deviation of Original Data (Predicted Concentration):")
print(std_original_data)

print("Averaged Test Data (Predicted Concentration):")
print(avg_test_data)
print("Standard Deviation of Test Data (Predicted Concentration):")
print(std_test_data)

# Combine means and standard deviations and export the results to CSV
avg_std_original = avg_original_data.copy()
avg_std_original["Predicted_Concentration_std"] = std_original_data["Predicted_Concentration"]

avg_std_test = avg_test_data.copy()
avg_std_test["Predicted_Concentration_std"] = std_test_data["Predicted_Concentration"]

test_output_path = output_file("copper_regression_results.csv", "multivariate_regression")
avg_std_test.to_csv(test_output_path, index=False)
print(f"Averaged and standard deviation results for testing data saved to: {test_output_path}")

# Create a grid at conductivity values 3000, 3250, 3500, 3750, and 4000 for the regression surface
conductivity_values = [3000, 3250, 3500, 3750, 4000]
x_mesh_new, y_mesh_new = np.meshgrid(
    conductivity_values,
    np.linspace(new_data["Cu-327nm_predict"].min(), new_data["Cu-327nm_predict"].max(), 10),
)
surface_features = pd.DataFrame(
    {"Conductivity": x_mesh_new.ravel(), "Cu-327nm_predict": y_mesh_new.ravel()}
)
z_mesh_new = regressor_new.predict(surface_features).reshape(x_mesh_new.shape)

# Create the three-dimensional plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot the averaged training observations
ax.scatter(
    avg_original_data["Conductivity"],
    avg_original_data["Cu-327nm_predict"],
    avg_original_data["Cu"],
    c="red",
    label="Training Data",
    alpha=0.6,
    s=100,
    marker="s",
)

# Plot the averaged testing observations
ax.scatter(
    avg_test_data["Conductivity"],
    avg_test_data["Cu-327nm_predict"],
    avg_test_data["Cu"],
    c="blue",
    label="Testing Data",
    alpha=0.6,
    s=100,
    marker="o",
)

# Plot the regression surface in gray
ax.plot_surface(x_mesh_new, y_mesh_new, z_mesh_new, color="grey", alpha=0.5)

# Set the horizontal-axis limits and tick labels
ax.set_xlim(3000, 4000)
ax.set_xticks([3000, 3250, 3500, 3750, 4000])
ax.set_xticklabels(["3000", "3250", "3500", "3750", "4000"])

# Set axis labels and font sizes
ax.set_xlabel("Conductivity (μS/cm)", fontsize=18, labelpad=10)
ax.set_ylabel("Cu Intensity", fontsize=18, labelpad=10)
ax.set_zlabel("Predicted Concentration", fontsize=18, labelpad=10)

# Set tick-label font sizes
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
ax.tick_params(axis="z", labelsize=14)

# Set the legend font size
ax.legend(loc="upper left", fontsize=16)

# Display the figure
plt.show()

# Calculate MAPE on the testing observations
actual_concentration_test = new_test_data["Cu"].values
predicted_concentration_test = new_test_data["Predicted_Concentration"].values
mape_test = (
    np.mean(
        np.abs(
            (actual_concentration_test - predicted_concentration_test) / actual_concentration_test
        )
    )
    * 100
)
print(f"MAPE for testing data: {mape_test:.2f}%")

# -----------------------------
#   Calculate R-squared to summarize the regression fit
# -----------------------------
# 1. Calculate R-squared on the training observations
y_train_pred = regressor_new.predict(X_train_new)
r2_train = r2_score(y_train_new, y_train_pred)
print(f"R² (Training Data): {r2_train:.4f}")

# 2. Calculate R-squared on the testing observations
y_test_true = new_test_data["Cu"].values
y_test_pred = new_test_data["Predicted_Concentration"].values
r2_test = r2_score(y_test_true, y_test_pred)
print(f"R² (Testing Data): {r2_test:.4f}")
