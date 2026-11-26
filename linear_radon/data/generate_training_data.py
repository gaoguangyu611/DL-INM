"""Generate linear CT IN trajectories, saving the raw Newton step and line-search factor separately.

The dataset stores:
  xk             : ground truth images, flattened to (d*d, 1)
  ydelta         : noisy sinograms, flattened to (n_measurements, 1)
  xkdelta        : IN states before each outer update, (d*d, 1)
  sk             : *unscaled* regularized Newton steps, (d*d, 1)
  gamma          : line-search relaxation coefficients, shape (N,)
  residual_norm  : pre-update measurement residual norms, shape (N,)
  delta          : noise levels, shape (N,)

IMPORTANT: The key 'gamma' here is the outer relaxation coefficient alpha_k in
Eq. (2.2) of the paper. It is NOT the neural-output projection scaling factor
called gamma(x,s) in Section 4.1.1.

Given a dataset row, the IN physical update is x_next = xkdelta + gamma * sk.
The full DL-INM update can later use x_next + beta * U_theta(xkdelta, sk),
where the network takes the raw sk (not gamma * sk).
"""

from pathlib import Path
import random

import h5py
import numpy as np
from tqdm import tqdm

from data.tikhonov import compute_sk


IMAGE_SIZE = 128
TAU = 1.5
MAX_OUTER = 50
SIGMA0 = 8.0
P = 0.8
MU_K = 0.8

# These are the outer line-search candidates, *not* the neural projection factor.
# Explicitly include 0.125, which was previously reached after the last failed
# trial but was never actually checked.
LINE_SEARCH_GAMMAS = (0.875, 0.750, 0.625, 0.500, 0.375, 0.250, 0.125)


def create_dataset(image_indices, dataset_path, is_training=True):
    """Save one row for every valid pre-update IN state (no post-stop extra row)."""
    R = np.load("R_limited.npy")
    with h5py.File("data/randshepp.mat", "r") as f:
        all_images = f["data"][:].transpose([-1, 0, 1])
    images = all_images[image_indices, :, :]

    xk_all = []              # ground truth
    ydelta_all = []          # noisy data
    xkdelta_all = []         # current iterate
    sk_all = []              # unscaled Newton step
    gamma_all = []           # outer relaxation factor; paper's alpha_k
    residual_norm_all = []   # optional: beta_k calculation for later training
    delta_all = []           # optional: record noise for diagnostics

    description = "training" if is_training else "validation"
    for idx_local, idx_global in enumerate(
        tqdm(image_indices, desc=f"Generating {description} dataset")
    ):
        image = np.asarray(images[idx_local], dtype=np.float64)
        image_max = np.max(image)
        if image_max <= 0:
            raise ValueError(f"Image {idx_global}: max intensity must be positive.")
        gt_1d = (image / image_max).reshape(-1, 1)
        if gt_1d.shape[0] != IMAGE_SIZE * IMAGE_SIZE:
            raise ValueError(f"Unexpected image shape for image {idx_global}: {image.shape}")

        y_exact = R @ gt_1d
        noise = 0.02 * np.max(np.abs(y_exact)) * np.random.randn(*y_exact.shape)
        y_delta = y_exact + noise
        delta = float(np.linalg.norm(noise))
        stop_threshold = TAU * delta

        x_k = np.full((IMAGE_SIZE * IMAGE_SIZE, 1), 0.01, dtype=np.float64)
        residual_norm = float(np.linalg.norm(R @ x_k - y_delta))
        print(f"sample {idx_global}, step 0: residual={residual_norm:.4e}")

        for k in range(MAX_OUTER):
            # Check *before* computing the Newton step, as in the paper's Algorithm 2.
            if residual_norm <= stop_threshold:
                break

            # compute_sk returns the inner regularization parameter as its
            # second output; this is NOT the outer relaxation coefficient.
            s_raw, _inner_regularization_param, _inner_count = compute_sk(
                x_k, R, y_delta,
                alpha_start=SIGMA0, q=P, mu_k=MU_K,
            )
            s_raw = np.asarray(s_raw).reshape(-1, 1)

            # Line search for the OUTER relaxation coefficient gamma (alpha_k).
            gamma = None
            next_residual = None
            for candidate in LINE_SEARCH_GAMMAS:
                x_trial = x_k + candidate * s_raw
                res_trial = float(np.linalg.norm(R @ x_trial - y_delta))
                if res_trial < residual_norm:
                    gamma = float(candidate)
                    next_residual = res_trial
                    break

            if gamma is None:
                print(
                    f"Warning: image {idx_global}, iteration {k}: "
                    "no tested relaxation coefficient decreases the residual; "
                    "stop this trajectory without adding an invalid row."
                )
                break

            # IMPORTANT: save RAW sk and gamma separately.
            # Do NOT overwrite s_raw with gamma*s_raw.
            xk_all.append(gt_1d.copy())
            ydelta_all.append(y_delta.copy())
            xkdelta_all.append(x_k.copy())
            sk_all.append(s_raw.copy())
            gamma_all.append(gamma)
            residual_norm_all.append(residual_norm)
            delta_all.append(delta)

            # Only the physical IN trajectory update uses the relaxation factor.
            x_k = x_k + gamma * s_raw
            residual_norm = next_residual
            print(
                f"sample {idx_global}, step {k+1}: "
                f"gamma={gamma:.3f}, residual={residual_norm:.4e}, "
                f"target={stop_threshold:.4e}"
            )

    if not sk_all:
        raise RuntimeError("No valid training states were generated.")

    xk_arr = np.stack(xk_all)
    ydelta_arr = np.stack(ydelta_all)
    xkdelta_arr = np.stack(xkdelta_all)
    sk_arr = np.stack(sk_all)
    gamma_arr = np.asarray(gamma_all, dtype=np.float64)
    residual_arr = np.asarray(residual_norm_all, dtype=np.float64)
    delta_arr = np.asarray(delta_all, dtype=np.float64)

    assert len(xk_arr) == len(ydelta_arr) == len(xkdelta_arr) == len(sk_arr) == len(gamma_arr)
    assert np.all(np.isin(gamma_arr, LINE_SEARCH_GAMMAS))
    assert np.all(residual_arr > TAU * delta_arr)

    Path(dataset_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dataset_path,
        xk=xk_arr,
        ydelta=ydelta_arr,
        xkdelta=xkdelta_arr,
        sk=sk_arr,                    # RAW Newton step
        gamma=gamma_arr,              # OUTER relaxation factor
        residual_norm=residual_arr,   # pre-update residual
        delta=delta_arr,              # norm(noise)
    )
    print(f"Saved {len(gamma_arr)} valid states to {dataset_path}")
    return xk_arr, ydelta_arr, xkdelta_arr, sk_arr, gamma_arr


def load_dataset(dataset_path):
    """Load the separate raw step / line-search coefficients."""
    with np.load(dataset_path) as data:
        return tuple(data[key].copy() for key in
                     ("xk", "ydelta", "xkdelta", "sk", "gamma"))


if __name__ == "__main__":
    total_samples = 120
    train_samples = 100
    val_samples = 20
    if train_samples + val_samples != total_samples:
        raise ValueError("Training and validation counts must sum to total_samples.")

    # IMPORTANT: set the seed BEFORE shuffling.
    random.seed(42)
    all_indices = list(range(total_samples))
    random.shuffle(all_indices)
    train_indices = all_indices[:train_samples]
    val_indices = all_indices[train_samples:]

    np.random.seed(42)
    create_dataset(train_indices, "data/train_dataset.npz", is_training=True)

    np.random.seed(999)
    create_dataset(val_indices, "data/val_dataset.npz", is_training=False)

    with np.load("data/train_dataset.npz") as train_data,\
         np.load("data/val_dataset.npz") as val_data:
        for name, data in (("Training", train_data), ("Validation", val_data)):
            print(f"{name}: {len(data['sk'])} states")
            print(f"  raw sk shape: {data['sk'].shape}")
            print(f"  gamma shape:  {data['gamma'].shape}")
            print(f"  physical update row 0 = xkdelta[0] + gamma[0] * sk[0]")
