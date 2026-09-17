# Reproduction guide

## Supported entry points

Start with `scripts/multivariate_regression.py` for calibration using the included observations. The published notebooks cover ANN/CNN/Transformer regression, spectral occlusion, and wastewater evaluation. Start Jupyter from the repository root and run the configuration cell before the experiment.

`python scripts/check_models.py --notebooks` loads all three saved checkpoints and executes each public notebook through its model-setup section, followed by finite-output and one-batch optimizer checks. It stops before the original training loop. The check uses synthetic inputs for model smoke tests and the supplied CSV files for notebook setup. The original long training schedules are not executed.

The checked environment uses Python 3.12.14, TensorFlow 2.16.2, and `tf-keras` 2.16.0. Notebook configuration selects `TF_USE_LEGACY_KERAS=1` before TensorFlow import. Mixing standalone Keras 3 layers with legacy models is unsupported; the notebooks consistently import layers through `tensorflow.keras`.

## Paths and output handling

`research_paths.py` resolves inputs relative to the repository. Optional environment variables select alternative data or output roots:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PLASMA_DATA_DIR` | `data/` | Spectral and calibration CSV tables |
| `PLASMA_ARTIFACT_DIR` | `artifacts/` | Original checkpoints and histories |
| `PLASMA_OUTPUT_DIR` | `outputs/` | Generated per-experiment results |

Generated models and histories are separated by experiment and cannot overwrite the supplied checkpoints. Repeating a save within the same experiment may replace that generated result; select a fresh `PLASMA_OUTPUT_DIR` to retain multiple runs.

## Data and research assumptions

The original numerical CSV, NPY, and H5 files are retained byte-for-byte. The data inventory records their schemas and hashes. Standard raw tables contain 1,912 columns, and most models select 1,862 wavelength features at positions `18:1880`. Averaged tables use a different metadata offset. Preserve wavelength order, metadata columns, and per-experiment masks.

The combined 720-row training and 600-row testing tables overlap with their long/short-duration component tables. They are not additional independent observations. Most notebooks clip at 60,000 and normalize by 60,000; selected variants mask spectral regions belonging to other elements.

Several historical experiments monitor a dataset named testing during training or use it for validation. Splits and calibration protocols are retained rather than relabeled as untouched holdout evaluation. The loss called SMAPE uses absolute error for a zero target and the original symmetric branch otherwise. Some plot scripts use fixed archived comparison values.

## Source corrections and preserved variants

- The nickel notebooks (`ann_nickel`, `cnn_nickel`, `tct_nickel_occlusion`, `tct_nickel_wastewater`) now clip their testing array before normalization; the source accidentally clipped the already-clipped training array in that cell, leaving saturated testing pixels (about 5-8 % of values) above the 60,000-count ceiling used for training. The duplicated seed-setting blocks were also collapsed into one block per notebook without changing the seeds.
- The lead Transformer notebook consistently uses its constructed `TCT_Model` for callbacks, fitting, and saving instead of an undefined `ANN_Model`.
- Optional tail sections requiring absent `test_long.csv` or `test_short.csv` acquisitions are retained in the private source archive. Public notebooks finish after their supplied-data experiment.
- A copper CNN and an online copper-occlusion variant require a separate absent online acquisition. Their full copies remain private.
- Three transfer notebooks cannot load the available shared checkpoint into their exact architecture (`axes do not match array`). The original source and checkpoints are preserved, but these variants are not supplied as runnable entry points.
- The unfinished time-series forecasting prototype remains private. Its decoder-output handling is not a validated spectral regression implementation.

No measurement arrays, checkpoint weights, or historical metrics were regenerated during cleanup. The source provenance ledger describes the original materials, including archived variants; the current notebook catalog lists public entry points.

## Validation limits

All three Keras checkpoints identify Keras 2.6.0 in their metadata and now pass loading and finite inference in the documented legacy runtime. That check does not establish compatibility with every historical transfer architecture or reproduce original training.

The repository validator checks syntax, cleared notebook outputs, English text and filenames, and numerical-input hashes. Notebook smoke checks cover setup, forward computation, and a single optimizer step. Complete training, SHAP/Grad-CAM runtime, accuracy, experimental timing, and hardware measurement were not reproduced.
