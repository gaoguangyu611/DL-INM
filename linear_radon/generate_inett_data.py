import h5py
import numpy as np
import os
from tqdm import tqdm
from data.tikhonov import compute_sk

def generate_sample_data(idx, images, R, tau=3):
    """提取一张图并运行迭代算法生成带伪影的重构解"""
    d = 128
    # 生成真解图像
    Ipgt = images[idx, :, :] / np.max(images[idx, :, :])
    gt_1d = Ipgt.reshape(-1, 1)

    # 物理观测
    y_exact = R @ gt_1d
    noise = 0.02 * np.max(np.abs(y_exact)) * np.random.randn(*y_exact.shape)
    y_delta = y_exact + noise

    # 偏差原理参数
    delta = np.linalg.norm(noise)
    stop_threshold = tau * delta

    # 迭代重构 (INM)
    x_k = np.zeros((d*d, 1))
    residual_norm = np.linalg.norm(R @ x_k - y_delta)

    for k in range(50):
        s_k, _ = compute_sk(x_k, R, y_delta, alpha_start=8, q=0.8, mu_k=0.8)
        
        # 线搜索
        gamma = 0.875
        for _ in range(6):
            xk_trial = x_k + s_k * gamma
            res_trial = np.linalg.norm(R @ xk_trial - y_delta)
            if res_trial < residual_norm:
                break
            gamma -= 0.125
        
        x_k = x_k + s_k * gamma
        residual_norm = np.linalg.norm(R @ x_k - y_delta)
        if residual_norm <= stop_threshold:
            break
            
    return gt_1d, x_k

def create_stratified_datasets(num_train=100, num_val=20, clean_ratio=0.2):
    # 计算各组所需的具体数量
    train_clean_count = int(num_train * clean_ratio)        # 20
    train_artifact_count = num_train - train_clean_count    # 80
    
    val_clean_count = int(num_val * clean_ratio)            # 4
    val_artifact_count = num_val - val_clean_count          # 16

    total_samples = num_train + num_val
    
    # 加载物理算子和原始图像
    R = np.load('R_limited.npy')
    f = h5py.File("data/randshepp.mat", 'r')
    all_images = f["data"][:].transpose([-1, 0, 1])
    images = all_images[range(total_samples), :, :]
    f.close()

    # 数据容器
    train_xk, train_xkdelta = [], []
    val_xk, val_xkdelta = [], []

    # 1. 生成带伪影的数据 (Artifact-heavy)
    # 总共需要 80 + 16 = 96 个
    print(f"正在生成带伪影数据 (共 {train_artifact_count + val_artifact_count} 个)...")
    artifact_indices = range(train_artifact_count + val_artifact_count)
    
    for i, idx in enumerate(tqdm(artifact_indices)):
        gt, recon = generate_sample_data(idx, images, R)
        if i < train_artifact_count:
            train_xk.append(gt)
            train_xkdelta.append(recon)
        else:
            val_xk.append(gt)
            val_xkdelta.append(recon)

    # 2. 生成 Clean 数据 (GT = Reconstruction)
    # 使用后续的索引，总共需要 20 + 4 = 24 个
    print(f"正在生成 Clean 数据 (共 {train_clean_count + val_clean_count} 个)...")
    clean_start_idx = train_artifact_count + val_artifact_count
    clean_indices = range(clean_start_idx, clean_start_idx + train_clean_count + val_clean_count)
    
    for i, idx in enumerate(tqdm(clean_indices)):
        Ipgt = images[idx, :, :] / np.max(images[idx, :, :])
        gt_1d = Ipgt.reshape(-1, 1)
        if i < train_clean_count:
            train_xk.append(gt_1d)
            train_xkdelta.append(gt_1d.copy())
        else:
            val_xk.append(gt_1d)
            val_xkdelta.append(gt_1d.copy())

    # 保存训练集
    np.savez_compressed("data/train_iNETT_dataset.npz", 
                        xk=np.array(train_xk), 
                        xkdelta=np.array(train_xkdelta))
    
    # 保存验证集
    np.savez_compressed("data/val_iNETT_dataset.npz", 
                        xk=np.array(val_xk), 
                        xkdelta=np.array(val_xkdelta))

    print("\n--- 数据生成完成 ---")
    print(f"训练集: {len(train_xk)} 个 (Clean: {train_clean_count}, Artifact: {train_artifact_count})")
    print(f"验证集: {len(val_xk)} 个 (Clean: {val_clean_count}, Artifact: {val_artifact_count})")

if __name__ == "__main__":
    np.random.seed(42)
    os.makedirs("data", exist_ok=True)
    create_stratified_datasets(num_train=160, num_val=40, clean_ratio=0.2)