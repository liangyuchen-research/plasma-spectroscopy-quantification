# Plasma Spectroscopy for Heavy Metal Quantification

Research code for estimating dissolved metal concentrations from optical emission spectra of plasmas in liquids. The collection compares dense neural networks, one-dimensional convolutional networks, and a convolutional Transformer, with spectral occlusion analysis, transfer-learning experiments, and wastewater evaluation.

The code combines machine learning with experimental spectroscopy: emission spectra are paired with concentration labels and acquisition metadata, then used to investigate prediction behavior across metal targets and solution conditions.

## Research components

- **Spectral regression:** TensorFlow/Keras ANN, CNN, and TCT experiments for copper, nickel, lead, and zinc. The TCT model combines one-dimensional convolutions with multi-head self-attention.
- **Model interpretation:** sliding-window spectral occlusion, importance-weighted spectral visualizations, and copper experiments with Grad-CAM and SHAP.
- **Transfer and wastewater experiments:** checkpoint-based fine-tuning for additional metal targets and evaluation on separately collected solution datasets.
- **Classical baseline:** multivariate linear regression using conductivity and characteristic-line intensity estimates, plus concentration recovery and relative-error plots.

This is a curated research snapshot. Algorithms, numerical datasets, and original experiment settings are preserved. Notebook outputs were cleared, comments were translated into English, and machine-specific paths were replaced with configurable repository paths. No new model performance claims are introduced.

## Repository layout

```text
notebooks/
  baselines/       ANN, CNN, and TCT comparison notebooks
  explainability/  Spectral occlusion, Grad-CAM, and SHAP experiments
  transfer/        Checkpoint-based metal-target transfer experiments
  wastewater/      Experiments using recollected and wastewater datasets
  experimental/    Exploratory time-series Transformer prototype
scripts/           Regression analysis, plotting, and lightweight validation
data/
  spectra/         Supplied spectral tables, including supplemental acquisitions
  regression/      Regression observations and original result tables
artifacts/
  checkpoints/     Three original Keras H5 checkpoints
  training_histories/  Original numerical NumPy arrays
docs/              Data inventory, source hashes, and reproduction notes
research_paths.py  Input resolution and per-experiment output locations
```

## Quick start

Use a dedicated Python 3.10 or 3.11 environment. The TensorFlow 2.15/Keras 2.15 requirements are a compatibility starting point for the APIs in the notebooks, not a recovered lockfile from the original experiments.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python scripts/validate_repository.py
python scripts/inspect_datasets.py
python -m jupyterlab
```

Activate the environment before installing packages. Start JupyterLab from the repository root and select an individual notebook. Run the introductory configuration cell first. Review the training cell before execution: several preserved experiments allow tens of thousands of epochs.

For a small non-neural baseline:

```bash
python scripts/multivariate_regression.py
python scripts/plot_regression_relative_error.py
```

The first script writes a regenerated CSV under `outputs/multivariate_regression/`; the second reproduces the original hard-coded comparison plot. Plotting requires a graphical backend or an explicitly configured noninteractive Matplotlib backend.

## Data and reproducibility

The supplied standard spectral tables contain 720 training rows and 600 testing rows across both acquisition-duration groups. These combined tables overlap with the separate long- and short-duration files; they are not independent additional observations. Most notebooks use the 1,862 wavelength features at positional columns `18:1880`, together with the selected metal concentration label. The smaller averaged tables use a different schema and are not drop-in replacements.

See [the reproduction guide](docs/reproduction.md) for external inputs, preprocessing assumptions, environment details, and known limitations. [The data inventory](docs/data-inventory.json) records exact row counts, sizes, and SHA-256 hashes. [The notebook catalog](docs/notebooks.json) identifies the notebooks that require additional acquisitions.

The source labels TCT as `Temporal_Convolutional_Transformer`. ORSFE is retained as the original method identifier; this repository does not invent an expansion for that acronym. The implemented operation is described directly as spectral occlusion analysis.

## Provenance and use

The repository was prepared from the research files supplied by the author, with three exact-name supplemental datasets recovered from the related research directory. A complete private archive retains every original file, including the duplicate notebook and all original notebook outputs. [The provenance ledger](docs/provenance.json) records source hashes without exposing local workstation paths.

No software or dataset license was present in the supplied snapshot. No new license grant is asserted here. Existing code and scientific dependencies retain their respective rights and attribution. Publication metadata and a formal citation should be added only when verified against the final paper.
