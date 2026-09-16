# Original model artifacts

The three H5 checkpoints and nine NumPy history arrays are exact copies of the supplied research artifacts. Their hashes appear in `docs/provenance.json`.

| Checkpoint | Stored model name | Saved Keras version |
| --- | --- | --- |
| `checkpoints/Cu_ANN.h5` | `ANN_Model` | 2.6.0 |
| `checkpoints/Cu_CNN.h5` | `CNN_Model` | 2.6.0 |
| `checkpoints/Cu_Transfer.h5` | `Temporal_Convolutional_Transformer` | 2.6.0 |

The NumPy arrays have four rows. In the source training code, these rows are assigned training loss, training MAE, validation loss, and validation MAE, respectively. Array lengths differ by experiment and should not be interpreted as directly comparable training budgets without the corresponding notebook settings.

All arrays were opened with `allow_pickle=False` during preparation. The model files were inspected as HDF5 metadata and loaded in TensorFlow 2.16.2 with legacy Keras 2.16. All three checkpoints produced finite predictions with their stored architectures. Numerical agreement with the original research results and compatibility with historical transfer variants are not established.

New runs write to `outputs/<experiment>/`, leaving these supplied artifacts unchanged. Checkpoint filenames alone do not identify the complete training data, seed, or originating notebook variant.
