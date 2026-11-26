import os
import numpy as np
import matplotlib.path as mpath
import random
from tqdm import tqdm
from data.tikhonov import compute_sk


def create_circular_mask(shape):
    """创建单位圆掩模，单位圆内为 True，圆外为 False"""
    h, w = shape
    radius = min(h, w) / 2
    center = (w / 2, h / 2)

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def generate_ground_truth(d):
    """
    生成自定义真解图像：
    背景为 0，单位圆内随机放置 2-3 个异常体，只包含三角形、正方形、圆形。
    """
    Ipgt = np.zeros((d, d))
    mask = create_circular_mask((d, d))

    num_anomalies = random.randint(2, 3)

    for _ in range(num_anomalies):
        anomaly_type = random.choice(["circle", "triangle", "square"])
        size = random.uniform(0.1, 0.2) * d
        value = random.uniform(0.6, 0.9)

        while True:
            x = random.randint(int(0.2 * d), int(0.8 * d))
            y = random.randint(int(0.2 * d), int(0.8 * d))
            if mask[y, x]:
                break

        if anomaly_type == "circle":
            radius = size / 2
            for i in range(int(y - radius), int(y + radius)):
                for j in range(int(x - radius), int(x + radius)):
                    if 0 <= i < d and 0 <= j < d and mask[i, j]:
                        if (i - y) ** 2 + (j - x) ** 2 <= radius ** 2:
                            Ipgt[i, j] = max(Ipgt[i, j], value)

        elif anomaly_type == "square":
            half_size = size / 2
            for i in range(int(y - half_size), int(y + half_size)):
                for j in range(int(x - half_size), int(x + half_size)):
                    if 0 <= i < d and 0 <= j < d and mask[i, j]:
                        Ipgt[i, j] = max(Ipgt[i, j], value)

        elif anomaly_type == "triangle":
            height = size
            width = size * (3 ** 0.5) / 2
            points = np.array([
                [x, y - height / 2],
                [x - width / 2, y + height / 2],
                [x + width / 2, y + height / 2],
            ])

            path = mpath.Path(points)
            grid_x, grid_y = np.meshgrid(np.arange(d), np.arange(d))
            inside = path.contains_points(
                np.column_stack((grid_x.ravel(), grid_y.ravel()))
            ).reshape(d, d)

            Ipgt[inside & mask] = value

    return Ipgt


def max_row_residual(Fx_2d, y_2d):
    """
    按你原来的非线性实验设置，使用每个角度 detector 向量的最大行残差。
    """
    return max([
        np.linalg.norm(Fx_2d[i, :] - y_2d[i, :])
        for i in range(Fx_2d.shape[0])
    ])


def generate_noisy_data(gt_1d, R, h, k_det, sigma):
    """
    非线性前向模型:
        F(x) = (R x)^2

    噪声形式沿用你原来的设置:
        y_delta = y_exact + abs(N(0, sigma))
    """
    sinogram_true = R @ gt_1d
    y_exact = sinogram_true ** 2
    y_exact_2d = y_exact.reshape(h, k_det)

    noise_2d = np.abs(np.random.normal(0, sigma, y_exact_2d.shape))
    y_noise_2d = y_exact_2d + noise_2d
    y_delta = y_noise_2d.reshape(-1, 1)

    delta = max_row_residual(y_noise_2d, y_exact_2d)

    return y_delta, y_noise_2d, y_exact_2d, delta


def reconstruct_by_inexact_newton(
    gt_1d,
    R,
    y_delta,
    y_noise_2d,
    h,
    k_det,
    tau=3.0,
    max_iter=50,
    alpha_start=8,
    q=0.8,
    mu_k=0.8,
    verbose=False,
    sample_id=None,
):

    x_k = np.ones((128 * 128, 1)) * 0.01

    Fxk_2d = ((R @ x_k) ** 2).reshape(h, k_det)
    residual_norm = max_row_residual(Fxk_2d, y_noise_2d)

    stop_threshold = tau * max_row_residual(
        y_noise_2d,
        ((R @ gt_1d) ** 2).reshape(h, k_det)
    )

    if verbose:
        prefix = f"sample {sample_id}, " if sample_id is not None else ""
        print(f"{prefix}step 0: Residual Norm = {residual_norm:.4e}, Target = {stop_threshold:.4e}")

    num_steps = 0

    for k in range(max_iter):
        s_k, alpha = compute_sk(
            x_k,
            R,
            y_delta,
            alpha_start=alpha_start,
            q=q,
            mu_k=mu_k,
        )

        best_gamma = 0.0
        best_residual = residual_norm

        for gamma in [0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]:
            x_trial = x_k + gamma * s_k
            Fx_trial_2d = ((R @ x_trial) ** 2).reshape(h, k_det)
            res_trial = max_row_residual(Fx_trial_2d, y_noise_2d)

            if res_trial < best_residual:
                best_gamma = gamma
                best_residual = res_trial
                break

        if best_gamma == 0.0:
            if verbose:
                prefix = f"sample {sample_id}, " if sample_id is not None else ""
                print(f"{prefix}step {k + 1}: no descent step found, stop.")
            break

        x_k = x_k + best_gamma * s_k
        residual_norm = best_residual
        num_steps = k + 1

        if verbose:
            prefix = f"sample {sample_id}, " if sample_id is not None else ""
            print(
                f"{prefix}step {k + 1}: "
                f"Residual Norm = {residual_norm:.4e}, "
                f"Target = {stop_threshold:.4e}, "
                f"gamma = {best_gamma:.3f}"
            )

        if residual_norm <= stop_threshold:
            break

    return x_k.copy(), residual_norm, stop_threshold, num_steps


def create_nett_dataset(
    num_samples,
    dataset_path,
    clean_ratio=0.2,
    seed=42,
    d=128,
    tau=3.0,
    sigma=10.0,
    max_iter=50,
    verbose=False,
):

    assert 0.0 <= clean_ratio < 1.0, "clean_ratio should be in [0, 1)."

    random.seed(seed)
    np.random.seed(seed)

    theta = np.linspace(0, 135, 60)
    h = len(theta)
    k_det = d

    R = np.load("R_limited.npy")

    num_clean = int(round(num_samples * clean_ratio))
    num_recon = num_samples - num_clean

    print("=" * 70)
    print(f"Creating dataset: {dataset_path}")
    print(f"Total samples: {num_samples}")
    print(f"Reconstruction samples: {num_recon} ({100 * num_recon / num_samples:.1f}%)")
    print(f"Clean-to-clean samples: {num_clean} ({100 * num_clean / num_samples:.1f}%)")
    print("=" * 70)

    xk_all = []
    xkdelta_all = []
    ydelta_all = []
    sample_type_all = []
    residual_all = []
    stop_threshold_all = []
    steps_all = []

    progress_bar = tqdm(total=num_samples, desc=f"Generating {os.path.basename(dataset_path)}")

    # ============================================================
    # 1. 生成 reconstruction-to-clean 样本，占 80%
    # ============================================================
    for idx in range(num_recon):
        Ipgt = generate_ground_truth(d)
        gt_1d = Ipgt.reshape(-1, 1)

        y_delta, y_noise_2d, y_exact_2d, delta = generate_noisy_data(
            gt_1d=gt_1d,
            R=R,
            h=h,
            k_det=k_det,
            sigma=sigma,
        )

        x_recon, residual_norm, stop_threshold, num_steps = reconstruct_by_inexact_newton(
            gt_1d=gt_1d,
            R=R,
            y_delta=y_delta,
            y_noise_2d=y_noise_2d,
            h=h,
            k_det=k_det,
            tau=tau,
            max_iter=max_iter,
            verbose=verbose,
            sample_id=idx + 1,
        )

        xk_all.append(gt_1d)
        xkdelta_all.append(x_recon)
        ydelta_all.append(y_delta)
        sample_type_all.append(0)  # 0 表示 reconstruction-to-clean
        residual_all.append(residual_norm)
        stop_threshold_all.append(stop_threshold)
        steps_all.append(num_steps)

        progress_bar.update(1)

    # ============================================================
    # 2. 生成 clean-to-clean 样本，占 20%
    # ============================================================
    for idx in range(num_clean):
        Ipgt = generate_ground_truth(d)
        gt_1d = Ipgt.reshape(-1, 1)

        # 为了保存格式统一，这里也生成对应的 y_delta。
        # 训练 NETT 去噪网络时可以不用 ydelta。
        y_delta, y_noise_2d, y_exact_2d, delta = generate_noisy_data(
            gt_1d=gt_1d,
            R=R,
            h=h,
            k_det=k_det,
            sigma=sigma,
        )

        xk_all.append(gt_1d)
        xkdelta_all.append(gt_1d.copy())
        ydelta_all.append(y_delta)
        sample_type_all.append(1)  # 1 表示 clean-to-clean
        residual_all.append(0.0)
        stop_threshold_all.append(tau * delta)
        steps_all.append(0)

        progress_bar.update(1)

    progress_bar.close()

    # ============================================================
    # 3. 打乱样本顺序，避免 clean 样本全部堆在后面
    # ============================================================
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    xk_arr = np.array(xk_all, dtype=np.float32)[indices]
    xkdelta_arr = np.array(xkdelta_all, dtype=np.float32)[indices]
    ydelta_arr = np.array(ydelta_all, dtype=np.float32)[indices]
    sample_type_arr = np.array(sample_type_all, dtype=np.int64)[indices]
    residual_arr = np.array(residual_all, dtype=np.float32)[indices]
    stop_threshold_arr = np.array(stop_threshold_all, dtype=np.float32)[indices]
    steps_arr = np.array(steps_all, dtype=np.int64)[indices]

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    np.savez_compressed(
        dataset_path,
        xk=xk_arr,
        xkdelta=xkdelta_arr,
        ydelta=ydelta_arr,
        sample_type=sample_type_arr,
        residual=residual_arr,
        stop_threshold=stop_threshold_arr,
        steps=steps_arr,
    )

    print(f"Saved dataset to: {dataset_path}")
    print(f"xk shape: {xk_arr.shape}")
    print(f"xkdelta shape: {xkdelta_arr.shape}")
    print(f"ydelta shape: {ydelta_arr.shape}")
    print(f"clean ratio check: {np.mean(sample_type_arr == 1):.4f}")

    return xk_arr, xkdelta_arr, ydelta_arr, sample_type_arr


def load_dataset(dataset_path):
    """加载保存的数据集"""
    data = np.load(dataset_path)

    xk = data["xk"]
    xkdelta = data["xkdelta"]
    ydelta = data["ydelta"] if "ydelta" in data.files else None
    sample_type = data["sample_type"] if "sample_type" in data.files else None

    return xk, xkdelta, ydelta, sample_type


if __name__ == "__main__":
    # ============================================================
    # 你可以在这里调整训练集和测试集大小
    # ============================================================
    train_samples = 200
    test_samples = 50

    clean_ratio = 0.2

    train_path = "data/NETT_train_dataset.npz"
    test_path = "data/NETT_test_dataset.npz"

    # ============================================================
    # 1. 生成训练集
    # ============================================================
    create_nett_dataset(
        num_samples=train_samples,
        dataset_path=train_path,
        clean_ratio=clean_ratio,
        seed=42,
        d=128,
        tau=3.0,
        sigma=10.0,
        max_iter=50,
        verbose=False,
    )

    # ============================================================
    # 2. 生成测试集
    # ============================================================
    create_nett_dataset(
        num_samples=test_samples,
        dataset_path=test_path,
        clean_ratio=clean_ratio,
        seed=999,
        d=128,
        tau=3.0,
        sigma=10.0,
        max_iter=50,
        verbose=False,
    )

    print("=" * 70)
    print("所有 NETT 数据集生成完成！")

    train_data = np.load(train_path)
    test_data = np.load(test_path)

    print(f"训练集样本数: {len(train_data['xk'])}")
    print(f"测试集样本数: {len(test_data['xk'])}")

    print(f"训练集 clean 比例: {np.mean(train_data['sample_type'] == 1):.4f}")
    print(f"测试集 clean 比例: {np.mean(test_data['sample_type'] == 1):.4f}")

    print(f"训练集 xk shape: {train_data['xk'].shape}")
    print(f"训练集 xkdelta shape: {train_data['xkdelta'].shape}")
    print(f"测试集 xk shape: {test_data['xk'].shape}")
    print(f"测试集 xkdelta shape: {test_data['xkdelta'].shape}")