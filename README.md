# Plasma Spectroscopy for Heavy Metal Quantification

Machine learning for estimating dissolved metal concentrations from optical emission spectra of plasmas in liquids. The experiments compare dense neural networks, one-dimensional convolutional networks, and a convolutional Transformer for Cu, Ni, Pb, and Zn, with spectral occlusion analysis, transfer learning, and wastewater evaluation.

## Research components

- **Spectral regression:** ANN, CNN, and convolutional Transformer models implemented in TensorFlow/Keras.
- **Model interpretation:** sliding-window occlusion identifies influential wavelength regions. Copper experiments also include Grad-CAM and SHAP.
- **Transfer and wastewater evaluation:** checkpoint-based fine-tuning for additional metal targets and experiments on separately collected solution datasets.
- **Classical calibration:** multivariate linear regression using conductivity and characteristic-line intensity estimates.

## Repository guide

| Location | Contents |
| --- | --- |
| `notebooks/baselines/` | ANN, CNN, and convolutional Transformer experiments |
| `notebooks/explainability/` | Spectral occlusion, Grad-CAM, and SHAP |
| `notebooks/transfer/` | Checkpoint-based transfer between metal targets |
| `notebooks/wastewater/` | Recollected-sample and wastewater experiments |
| `notebooks/experimental/` | Exploratory time-series Transformer implementation |
| `scripts/` | Regression, plotting, dataset inspection, and validation |
| `data/` | Spectral tables, regression observations, and result tables |
| `artifacts/` | Three Keras checkpoints and nine training-history arrays |
| `docs/` | Experiment catalog, data inventory, and reproduction guide |

## Run the calibration baseline

Use Python 3.10 or 3.11. Create a virtual environment and activate it with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux.

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

Start JupyterLab from the repository root, open an individual notebook, and run its configuration cell first. Review the training settings before execution: some experiments permit tens of thousands of epochs. The requirements target TensorFlow/Keras 2.15 compatibility; the original experiments did not include a complete environment lockfile. Optional explanation and prototype dependencies are listed in `requirements-optional.txt`.

The standard combined spectral tables contain 720 training rows and 600 testing rows. They overlap with the separate long- and short-duration tables and must not be counted as additional independent observations. Most notebooks select 1,862 wavelength features from columns `18:1880`. Averaged tables use a different schema.

## Data and reproducibility

The calibration baseline has been run with the included data. Full neural training and checkpoint inference have not been rerun in the documented environment. Some experimental notebooks require additional acquisitions, and several original experiments use a testing dataset during training or validation. Their results require the corresponding protocol when interpreted.

The [reproduction guide](docs/reproduction.md) documents preprocessing, external inputs, checkpoint compatibility, and a corrected nickel testing-preprocessing assignment. The [data inventory](docs/data-inventory.json) records row counts and hashes, and the [notebook catalog](docs/notebooks.json) lists each experiment and its input requirements.

`TCT` is the original identifier for `Temporal_Convolutional_Transformer`. `ORSFE` is retained where it occurs in the source; the implemented operation is described here as spectral occlusion analysis.

## Data and code use

This repository contains the research implementation and associated numerical data. The [provenance ledger](docs/provenance.json) records source hashes, and the [artifact notes](artifacts/README.md) describe the saved checkpoints. Original numerical data and model artifacts are unchanged.

No software or dataset license is included. Contact the repository owner regarding reuse or access to additional acquisitions. Existing third-party dependencies retain their respective licenses.
