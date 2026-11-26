import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import random
from tqdm import tqdm
# from data.landweber_nonlinear import landweber_schl
from data.tikhonov import compute_sk


def create_dataset(image_indices, dataset_path, is_training=True):
    """创建真解和反演结果数据集"""
    # 参数设置
    d = 128
    tau = 1.1

    R = np.load('R_limited.npy')

    # 初始化数据集
    xk_all = []
    ydelta_all = []
    xkdelta_all = []
    sk_all = []

    # 创建进度条
    progress_bar = tqdm(total=len(image_indices), desc=f"Generating {'training' if is_training else 'validation'} dataset")
    all_images = h5py.File("data/randshepp.mat")["data"][:].transpose(
        [-1, 0, 1]
    )
    
    # 只提取指定索引的图像
    images = all_images[image_indices, :, :]

    for idx_local, idx_global in enumerate(image_indices):

        # 1. 生成自定义真解图像
        Ipgt = images[idx_local, :, :] / np.max(images[idx_local, :, :])     # d*d 图像
        gt_1d = Ipgt.reshape(-1, 1)

        # 计算真实投影
        y_exact = R @ gt_1d                           # (, 1)

        # 添加噪声 randn是高斯噪声 标准正态分布
        noise = 0.02 * np.max(np.abs(y_exact)) * np.random.randn(*y_exact.shape)
        y_delta = y_exact + noise

        # 计算噪声水平
        delta = np.linalg.norm(noise)
        stop_threshold = tau * delta

        x_k = np.zeros((128*128,1))
        Fxk = R @ x_k
        residual_norm = np.linalg.norm(Fxk - y_delta)
        print(f"sample {idx_global}, step {0}: Residual Norm = {residual_norm:.4e}")

        for k in range(50):
            
            s_k, alpha = compute_sk(x_k, R, y_delta, alpha_start=8, q=0.8, mu_k=0.8)
            
            # ---------------- 加入回溯线搜索 (Line Search) ----------------
            gamma = 0.875
            for _ in range(6):  # 0.125~0.875

                xk_trial = x_k + s_k * gamma
                Fxk_2d1 = R @ xk_trial
                res_trial = np.linalg.norm(Fxk_2d1 - y_delta)

                if res_trial < residual_norm:
                    break
                gamma += -0.125
            
            s_k = s_k * gamma

            # 只保存最终结果，不保存中间步骤
            x_k = x_k + s_k
            Fxk = R @ x_k
            residual_norm = np.linalg.norm(Fxk - y_delta)

            print(f"sample {idx_global}, step {k+1}: Residual Norm = {residual_norm:.4e}, Target = {stop_threshold:.4e}")
            
            # --- 检查提前停止 ---
            if residual_norm <= stop_threshold:
                # 达到偏差原理可以提前结束当前图像的轨迹
                break

        s_k, alpha = compute_sk(x_k, R, y_delta, alpha_start=8, q=0.8, mu_k=0.8)
            
        # ---------------- 加入回溯线搜索 (Line Search) ----------------
        gamma = 0.875
        for _ in range(6):  # 0.125~0.875

            xk_trial = x_k + s_k * gamma
            Fxk_2d1 = R @ xk_trial
            res_trial = np.linalg.norm(Fxk_2d1 - y_delta)

            if res_trial < residual_norm:
                break
            gamma += -0.125
            
        s_k = s_k * gamma

        # 保存最终结果
        xk_all.append(gt_1d)
        ydelta_all.append(y_delta)
        xkdelta_all.append(x_k.copy())
        progress_bar.update(1)

    # 将列表转换为numpy数组
    xk_arr = np.array(xk_all)
    ydelta_arr = np.array(ydelta_all)
    xkdelta_arr = np.array(xkdelta_all)

    # 保存数据集 - 只保存图像数据
    np.savez_compressed(
        dataset_path,
        xk=xk_arr,
        ydelta=ydelta_arr,
        xkdelta=xkdelta_arr
    )

    return xk_arr, ydelta_arr, xkdelta_arr


# 加载数据的辅助函数
def load_dataset(dataset_path):
    """加载保存的数据集"""
    data = np.load(dataset_path)
    xk = data['xk']
    ydelta = data['ydelta']
    xkdelta = data['xkdelta']

    return xk, ydelta, xkdelta


if __name__ == "__main__":
    
    d = 128
    

    total_samples = 200
    train_samples = 160  
    val_samples = 40  
    
    print(f"总共使用 {total_samples} 个图像")
    print(f"训练集: {train_samples} 个图像")
    print(f"验证集: {val_samples} 个图像")
    
    # 随机打乱索引
    all_indices = list(range(total_samples))
    random.shuffle(all_indices)
    
    # 分配训练集和验证集的索引
    train_indices = all_indices[:train_samples]
    val_indices = all_indices[train_samples:]
    
    # ========================================================
    # 1. 生成训练集 (Train Set)
    # ========================================================
    # 固定随机种子，确保训练集可重复
    random.seed(42)
    np.random.seed(42)
    print("开始生成训练集 (Train Dataset)...")
    create_dataset(
        image_indices=train_indices, 
        dataset_path="data/res_train_dataset.npz", 
        is_training=True
    )
    
    # ========================================================
    # 2. 生成验证集 (Validation Set)
    # ========================================================
    # 使用不同的随机种子
    random.seed(999)
    np.random.seed(999)
    print("开始生成验证集 (Validation Dataset)...")
    create_dataset(
        image_indices=val_indices, 
        dataset_path="data/res_val_dataset.npz", 
        is_training=False
    )
    
    print("所有数据集生成完成!")
    
    # 验证生成的数据集大小
    train_data = np.load("data/res_train_dataset.npz")
    val_data = np.load("data/res_val_dataset.npz")
    
    print(f"训练集样本数: {len(train_data['xk'])}")
    print(f"验证集样本数: {len(val_data['xk'])}")