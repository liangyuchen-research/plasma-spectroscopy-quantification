# Plasma Spectroscopy for Heavy Metal Quantification

Machine learning for estimating dissolved metal concentrations from optical emission spectra of plasmas in liquids. The experiments compare dense neural networks, one-dimensional convolutional networks, and a convolutional Transformer for Cu, Ni, Pb, and Zn, with spectral occlusion analysis, transfer learning, and wastewater evaluation.

## Publication

Chen, L.-Y., Wang, C.-Y., and Hsu, C.-C. Machine Learning-Based System for Online Quantitative Monitoring of Heavy Metals Across Different Aqueous Matrices Using Spectroscopy of Plasmas in Liquids. *Talanta* **297** (2026), 128652. [DOI: 10.1016/j.talanta.2025.128652](https://doi.org/10.1016/j.talanta.2025.128652).

The related [GAN spectral-restoration project](https://github.com/liangyuchen-research/conditional-gan-spectral-restoration) covers restoration before downstream quantification. The regression and occlusion-analysis implementations are maintained in this repository.

## Research components

- **Spectral regression:** ANN, CNN, and convolutional Transformer models implemented in TensorFlow/Keras.
- **Model interpretation:** sliding-window occlusion identifies influential wavelength regions. Copper experiments also include Grad-CAM and SHAP.
- **Wastewater evaluation:** regression experiments on separately collected solution datasets and archived ANN, CNN, and Transformer checkpoints.
- **Classical calibration:** multivariate linear regression using conductivity and characteristic-line intensity estimates.

## Repository guide

| Location | Contents |
| --- | --- |
| `notebooks/baselines/` | ANN, CNN, and convolutional Transformer experiments |
| `notebooks/explainability/` | Spectral occlusion, Grad-CAM, and SHAP |
| `notebooks/wastewater/` | Recollected-sample and wastewater experiments |
| `scripts/` | Regression, plotting, dataset inspection, and validation |
| `data/` | Spectral tables, regression observations, and result tables |
| `artifacts/` | Three Keras checkpoints and nine training-history arrays |
| `docs/` | Experiment catalog, data inventory, and reproduction guide |

## Run the calibration baseline

Use Python 3.10–3.12. Create a virtual environment and activate it with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux.

```bash
python -m venv .venv
# Activate the environment before continuing.
python -m pip install -r requirements-analysis.txt
python scripts/validate_repository.py
python scripts/inspect_datasets.py
python scripts/multivariate_regression.py
```

The regression script uses the included observations and exports concentration estimates to `outputs/multivariate_regression/copper_regression_results.csv`. It also displays the fitted calibration surface. For a headless session, set the `MPLBACKEND` environment variable to `Agg` before execution. `scripts/plot_regression_relative_error.py` displays a separate comparison based on recorded values.

## Neural experiments

```bash
python -m pip install -r requirements.txt
python -m jupyterlab
```

Start JupyterLab from the repository root, open an individual notebook, and run its configuration cell first. Review the training settings before execution: some experiments permit tens of thousands of epochs. The requirements use TensorFlow 2.16.2 with legacy Keras 2.16. Notebook setup cells select the legacy runtime before importing TensorFlow. Optional SHAP dependencies are listed in `requirements-optional.txt`.

The standard combined spectral tables contain 720 training rows and 600 testing rows. They overlap with the separate long- and short-duration tables and must not be counted as additional independent observations. Most notebooks select 1,862 wavelength features from columns `18:1880`. Averaged tables use a different schema.

## Data and reproducibility

The calibration baseline has been run with the included observations. The three archived checkpoints load and produce finite predictions. Every published notebook has passed supplied-data setup, model construction, and one bounded optimizer step without running its training loop.

```bash
python scripts/check_models.py --notebooks
```

These checks establish executable model setup, not reproduced scientific performance. Full neural training and the complete SHAP/Grad-CAM analyses have not been rerun. Several original experiments use a testing dataset during training or validation, so their results require the corresponding protocol when interpreted. Historical variants with unavailable acquisitions, incompatible transfer checkpoints, or an unfinished forecasting model remain in the private source archive.

The [reproduction guide](docs/reproduction.md) documents preprocessing, compatibility, validation scope, and source corrections. The [data inventory](docs/data-inventory.json) records row counts and hashes, and the [notebook catalog](docs/notebooks.json) lists each experiment and its input requirements.

`TCT` is the original identifier for `Temporal_Convolutional_Transformer`. `ORSFE` is retained where it occurs in the source; the implemented operation is described here as spectral occlusion analysis.

## Data and code use

This repository contains the research implementation and associated numerical data. The [provenance ledger](docs/provenance.json) records source hashes, and the [artifact notes](artifacts/README.md) describe the saved checkpoints. Original numerical data and model artifacts are unchanged.

No software or dataset license is included. Contact the repository owner regarding reuse or access to additional acquisitions. Existing third-party dependencies retain their respective licenses.
