# DL-INM: Deep-learning enhanced inexact Newton method

Implementation of **A deep-learning enhanced inexact Newton framework for nonlinear inverse problems**.

DL-INM combines a regularized Newton step with a learned correction from a dual-input U-Net. This repository considers two reconstruction problems on a `128 × 128` image grid:

- **Linear Radon tomography:** `y = R x`.
- **Nonlinear Schlieren tomography:** `y = (R x)**2`.

Comparison methods include inexact Newton (IN), Landweber iteration, IN-TV, residual-network corrections, and learned regularization methods.

## Installation

The reference environment uses **Python 3.8.19** and **PyTorch 2.3.1**. From the repository root:

```bash
conda create -n dl_inm python=3.8.19
conda activate dl_inm
python -m pip install -r requirements.txt
```

For CUDA execution, use a PyTorch build compatible with the local GPU driver. Training scripts select CUDA when available and otherwise use the CPU. The Python version is configured through the environment manager; it is not a pip dependency.

The dense float64 measurement matrix has shape `(7680, 16384)` and occupies approximately **0.94 GiB**. Data generation, reconstruction, and matrix pseudoinversion require additional memory and may take substantial time.

## Repository structure

```text
.
├── generate_operator.py
├── requirements.txt
├── linear_radon/
│   ├── train_dl_inm.py
│   ├── train_residual.py
│   ├── train_inett.py
│   ├── train_nett.py
│   ├── test_learning.py
│   ├── test_traditional.py
│   └── data/
│       ├── randshepp.mat
│       ├── generate_training_data.py
│       ├── generate_residual_data.py
│       ├── generate_inett_data.py
│       ├── tikhonov.py
│       └── landweber.py
└── nonlinear_schlieren/
    ├── train_dl_inm.py
    ├── train_residual.py
    ├── train_nett.py
    ├── test_learning.py
    ├── test_traditional.py
    └── data/
        ├── generate_training_data.py
        ├── generate_residual_data.py
        ├── generate_nett_data.py
        ├── tikhonov.py
        └── landweber.py
```

Run training and data-generation commands **from the corresponding experiment directory**. Input and output paths are relative to the working directory. Use `python -m data.<module>` for the data generators so their local imports resolve correctly.

## Generate the measurement matrix


```bash
python generate_operator.py --output linear_radon/R_limited.npy
python generate_operator.py --output nonlinear_schlieren/R_limited.npy
```

Both experiments use the same matrix. You can generate it once and copy it into the other experiment directory. The generator refuses to overwrite an existing output file.

## Linear Radon experiment

### Generate data and train DL-INM

The phantom bank `data/randshepp.mat` contains the HDF5 array `data` of shape `(128, 128, 2500)`. Images are normalized by their maximum intensity before projection.

```bash
cd linear_radon
python -m data.generate_training_data
python train_dl_inm.py
```

The generator splits the first 120 phantom indices into 100 training and 20 validation images. It saves one row for each accepted pre-update IN state, so the number of training rows differs from the number of phantom images. Observations use

```text
y_delta = R x + 0.02 max(abs(R x)) * N(0, 1).
```

The generated files are `data/train_dataset.npz` and `data/val_dataset.npz`:

| Field | Meaning |
|---|---|
| `xk` | Ground-truth image |
| `ydelta` | Noisy measurement |
| `xkdelta` | Current reconstruction before the outer update |
| `sk` | Raw regularized Newton step |
| `gamma` | Outer relaxation coefficient multiplying `sk` |
| `residual_norm` | Current measurement residual norm |
| `delta` | Noise norm |

The network receives the current reconstruction and the raw Newton step. The dataset field `gamma` represents the outer relaxation coefficient, not the scaling of the neural correction.

Training writes `models/best.pth` and `models/final_best.pth`.

### Train comparison networks

```bash
python -m data.generate_residual_data
python train_residual.py
python -m data.generate_inett_data
python train_inett.py
```

## Nonlinear Schlieren experiment

### Generate data and train DL-INM

From the repository root:

```bash
cd nonlinear_schlieren
python -m data.generate_training_data
python train_dl_inm.py
```

The generator constructs phantoms from circles, triangles, and squares within a circular support. It uses 100 training and 20 validation images with seeds 42 and 999, respectively. Measurements follow `y_delta = (R x)**2 + abs(N(0, 10**2))`.

The generated `data/train_dataset.npz` and `data/val_dataset.npz` contain `xk`, `ydelta`, `xkdelta`, and `sk`. Here, the stored `sk` already includes the outer relaxation factor. Use each experiment's data generator with its corresponding training script.

Training writes `models/best.pth` and `models/final_best.pth`. The nonlinear evaluation code expects `models/best.pth` for DL-INM.

### Train comparison networks

```bash
python -m data.generate_residual_data
python train_residual.py
python -m data.generate_nett_data
python train_nett.py
```

## Training configuration

Each `train_*.py` file defines a `Config` class for data paths, learning rate, weight decay, loss function, and device settings. The supplied configurations use 100 epochs and a batch size of 8. Training scripts also save loss plots to their configured output folders.

## Reconstruction and evaluation

`test_learning.py` compares learning-based methods. `test_traditional.py` compares DL-INM with IN, Landweber, and IN-TV. The scripts provide reconstruction, metric computation, and visualization routines. Metrics include relative error, PSNR, SSIM, iteration counts, and reconstruction time.

Once the input files are available, run the entry points from the relevant experiment directory:

```bash
python test_learning.py
python test_traditional.py
```

## Reproducibility

Record package versions, random seeds, training configurations, device information, and checkpoint selection for each experiment. Data-generation seeds are set in the generators; training randomness and GPU kernels can still affect results. Training settings are defined separately for each network, so use the corresponding configuration when reproducing a comparison.
