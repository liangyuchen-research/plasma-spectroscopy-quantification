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
from research_paths import data_file, external_file, checkpoint_file, history_file, output_file

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Conductivity values
conductivity = [3000, 3250, 3500, 3750, 4000]

# Recovery percentages for 7.5 ppm and 15 ppm
recovery_7_5 = [6.939, 6.776, 7.705, 7.356, 6.074]
# std_7_5 = [0.222563409,0.224096208,0.305934709,0.390535237,0.390780517]
recovery_15 = [10.073, 12.006, 14.634, 13.947, 14.065]
# std_15 = [0.347370904,1.220330114,0.79163668,0.554538327,0.727954113]

# The 100% reference value for comparison
reference = 7.5

# Calculate differences from 100%
difference_7_5 = [((val / reference) * 100) - 100 for val in recovery_7_5]
difference_15 = [((val / (2 * reference)) * 100) - 100 for val in recovery_15]

# Calculate error bars as relative error %
# error_7_5 = [(std / reference) * 100 for std in std_7_5]
# error_15 = [(std / (2 * reference)) * 100 for std in std_15]

# Plotting
x = np.arange(len(conductivity))  # label locations
width = 0.35  # width of the bars

fig, ax = plt.subplots(figsize=(8, 6))

# Bars for 7.5 ppm (solid blue)
bars1 = ax.bar(x - width / 2, difference_7_5, width, color="skyblue", label="7.5 ppm")

# Bars for 15 ppm (orange with hatch)
bars2 = ax.bar(
    x + width / 2,
    difference_15,
    width,
    color="orange",
    edgecolor="gray",
    hatch="////",
    label="15 ppm",
)

# Add a horizontal line for 0% difference
ax.axhline(0, color="black", linestyle="--", linewidth=1.4)

# Labels, and custom x-axis tick labels
ax.set_xlabel("Conductivity (μS/cm)", fontsize=20)
ax.set_ylabel("Relative Error (%)", fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels([f"{cond}" for cond in conductivity], fontsize=18)
ax.tick_params(axis="y", labelsize=18)

# Custom legend with hatch pattern
legend_patches = [
    mpatches.Patch(color="skyblue", label="7.5 ppm"),  # Blue solid
    mpatches.Patch(
        facecolor="orange", edgecolor="gray", hatch="////", label="15 ppm"
    ),  # Orange with gray hatch
]
ax.legend(handles=legend_patches, fontsize=18)

# Add grid for better readability
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Add border around the plot
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)

plt.tight_layout()
plt.show()
