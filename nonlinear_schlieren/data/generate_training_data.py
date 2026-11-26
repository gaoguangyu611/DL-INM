import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import random
from tqdm import tqdm
from data.tikhonov import compute_sk


def create_circular_mask(shape):
    """创建单位圆掩模，单位圆内为True，圆外为False"""
    h, w = shape
    radius = min(h, w) / 2
    center = (w / 2, h / 2)

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def generate_ground_truth(d):
    """生成自定义真解图像：背景为0，单位圆内随机放置2-3个异常体（只包含三角形、正方形、圆形）"""
    # 创建黑色背景
    Ipgt = np.zeros((d, d))

    # 创建圆形掩模
    mask = create_circular_mask((d, d))

    # 在单位圆内生成2-3个异常体
    num_anomalies = random.randint(2, 3)

    for _ in range(num_anomalies):
        # 随机选择异常体类型（不包括字母）
        anomaly_type = random.choice(['circle', 'triangle', 'square'])
        size = random.uniform(0.1, 0.2) * d  # 尺寸范围
        value = random.uniform(0.6, 0.9)  # 灰度值范围

        # 在单位圆内随机位置
        while True:
            x = random.randint(int(0.2 * d), int(0.8 * d))
            y = random.randint(int(0.2 * d), int(0.8 * d))
            if mask[y, x]:
                break

        if anomaly_type == 'circle':
            # 生成圆形异常体
            radius = size / 2
            for i in range(int(y - radius), int(y + radius)):
                for j in range(int(x - radius), int(x + radius)):
                    if i >= 0 and i < d and j >= 0 and j < d and mask[i, j]:
                        if (i - y) ** 2 + (j - x) ** 2 <= radius ** 2:
                            Ipgt[i, j] = max(Ipgt[i, j], value)

        elif anomaly_type == 'square':
            # 生成正方形异常体
            half_size = size / 2
            for i in range(int(y - half_size), int(y + half_size)):
                for j in range(int(x - half_size), int(x + half_size)):
                    if i >= 0 and i < d and j >= 0 and j < d and mask[i, j]:
                        Ipgt[i, j] = max(Ipgt[i, j], value)

        elif anomaly_type == 'triangle':
            # 生成三角形异常体
            height = size
            width = size * (3 ** 0.5) / 2  # 等边三角形
            points = np.array([
                [x, y - height / 2],  # 顶点
                [x - width / 2, y + height / 2],  # 左下角
                [x + width / 2, y + height / 2]  # 右下角
            ])
            path = mpath.Path(points)
            grid_x, grid_y = np.meshgrid(np.arange(d), np.arange(d))
            inside = path.contains_points(np.column_stack((grid_x.ravel(), grid_y.ravel()))).reshape(d, d)
            Ipgt[inside & mask] = value

    return Ipgt


def create_dataset(num_samples, dataset_path):
    """创建真解和反演结果数据集"""
    # 参数设置
    d = 128
    theta = np.linspace(0, 135, 60)
    # step_size = 1e-7
    # nitermax = 1000
    tau = 3
    sigma = 10.0  # 噪声标准差
    h = len(theta)
    k_det = d

    R = np.load('R_limited.npy')

    # 初始化数据集

    xk_all = []
    ydelta_all = []
    xkdelta_all = []
    sk_all = []

    # 创建进度条
    progress_bar = tqdm(total=num_samples, desc="Generating dataset")

    for idx in range(num_samples):

        # 1. 生成自定义真解图像
        Ipgt = generate_ground_truth(d)     # d*d 图像
        gt_1d = Ipgt.reshape(-1, 1)


        # 计算真实投影
        sinogramtrue = R @ gt_1d
        y_exact = sinogramtrue ** 2  # 投影数据的平方

        y_exact_2d = y_exact.reshape(h, k_det)

        # 添加噪声
        y_noise_2d = y_exact_2d + np.abs(np.random.normal(0, sigma, y_exact_2d.shape))
        y_delta = y_noise_2d.reshape(-1, 1)

        # 计算噪声水平
        delta = max([np.linalg.norm(y_noise_2d[i, :] - y_exact_2d[i, :]) for i in range(h)])
        stop_threshold = tau * delta

        x_k = np.ones((128*128,1)) * 0.01
        Fxk_2d = ((R @ x_k)**2).reshape(h, k_det)
        residual_norm = max([np.linalg.norm(Fxk_2d[i, :] - y_noise_2d[i, :]) for i in range(60)])

        print(f"step {0}: Residual Norm = {residual_norm:.4e}")
        

        for k in range(50):
          
            
            s_k, alpha = compute_sk(x_k, R, y_delta, alpha_start=8, q=0.8, mu_k=0.8)
            
            # ---------------- 加入回溯线搜索 (Line Search) ----------------
            gamma = 0.875
            for _ in range(6):  # 0.125~0.875

                xk_trial = x_k + s_k * gamma
                Fxk_2d1 = ((R @ xk_trial)**2).reshape(h, k_det)
                res_trial = max([np.linalg.norm(Fxk_2d1[i, :] - y_noise_2d[i, :]) for i in range(60)])
                # res_trial = np.linalg.norm((R @ xk_trial)**2 - y_delta)

                if res_trial < residual_norm:
                    # x_k = xk_trial
                    # residual_norm = res_trial
                    break
                gamma += -0.125

            
            s_k = s_k * gamma

            xk_all.append(gt_1d)
            ydelta_all.append(y_delta)
            # DL-INM
            xkdelta_all.append(x_k.copy())
            sk_all.append(s_k.copy())

            x_k = x_k + s_k
            Fxk_2d = ((R @ x_k)**2).reshape(h, k_det)
            residual_norm = max([np.linalg.norm(Fxk_2d[i, :] - y_noise_2d[i, :]) for i in range(60)])
            # residual_norm = np.linalg.norm((R @ x_k)**2 - y_delta)
            

            print(f"sample{idx+1},step {k+1}: Residual Norm = {residual_norm:.4e}, Target = {stop_threshold:.4e}")
            
            # --- 检查提前停止 ---
            if residual_norm <= stop_threshold:
                # 达到偏差原理可以提前结束当前图像的轨迹
                break
            
        s_k, alpha = compute_sk(x_k, R, y_delta, alpha_start=8, q=0.8, mu_k=0.8)
        # ---------------- 加入回溯线搜索 (Line Search) ----------------
        gamma = 0.875
        for _ in range(6):  # 0.125~0.875

            xk_trial = x_k + s_k * gamma
            Fxk_2d1 = ((R @ xk_trial)**2).reshape(h, k_det)
            res_trial = max([np.linalg.norm(Fxk_2d1[i, :] - y_noise_2d[i, :]) for i in range(60)])
            # res_trial = np.linalg.norm((R @ xk_trial)**2 - y_delta)

            if res_trial < residual_norm:
                # x_k = xk_trial
                # residual_norm = res_trial
                break
            gamma += -0.125
        s_k = s_k * gamma
            
        xk_all.append(gt_1d)
        ydelta_all.append(y_delta)
        # DL-INM
        xkdelta_all.append(x_k.copy())
        sk_all.append(np.zeros(s_k.shape))


        # 画个图看看最后一个采样的样本（验证程序正常）
        if idx == num_samples - 1:
            im_reshaped = x_k.reshape(d, d)
            plt.figure(figsize=(8, 8))
            plt.imshow(im_reshaped, cmap='viridis', vmin=im_reshaped.min(), vmax=im_reshaped.max())
            plt.title(f'IN Final State (Sample {idx}, Steps {k+1})')
            plt.colorbar()
            plt.axis('off')
            plt.savefig(f'data/im_IN_final_sample_{idx}.png', dpi=300, bbox_inches='tight')
            plt.close()

        progress_bar.update(1)

    # 将列表转换为numpy数组
    xk_arr = np.array(xk_all)
    ydelta_arr = np.array(ydelta_all)
    xkdelta_arr = np.array(xkdelta_all)
    sk_arr = np.array(sk_all)

    # 保存数据集 - 只保存图像数据
    np.savez_compressed(
        dataset_path,
        xk=xk_arr,
        ydelta=ydelta_arr,
        xkdelta=xkdelta_arr,
        sk=sk_arr
    )

    return xk_arr, ydelta_arr, xkdelta_arr, sk_arr

def plot_samples(d, ground_truths, reconstructions, num_samples=3):
    """随机选择几个样本并绘图"""
    num_total = len(ground_truths)
    indices = random.sample(range(num_total), num_samples)

    for idx, sample_idx in enumerate(indices):
        Ipgt = ground_truths[sample_idx].reshape(d, d)
        recon = reconstructions[sample_idx].reshape(d, d)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Sample {sample_idx}", fontsize=16)

        # 确定共同的色彩范围
        vmin = min(Ipgt.min(), recon.min())
        vmax = max(Ipgt.max(), recon.max())

        # 真解
        im0 = axes[0].imshow(Ipgt, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title('Ground Truth')
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # 重建结果
        im1 = axes[1].imshow(recon, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title('Reconstructed')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # 误差图
        error_map = np.abs(Ipgt - recon)
        im2 = axes[2].imshow(error_map, cmap='viridis')
        axes[2].set_title('Absolute Error')
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为标题留出空间
        plt.savefig(f'data/sample_plots/sample_{sample_idx}.png', dpi=300)
        plt.close()


# 加载数据的辅助函数
def load_dataset():
    """加载保存的数据集"""
    data = np.load("data/dataset.npz")
    xk = data['xk']
    ydelta = data['ydelta']
    xkdelta = data['xkdelta']
    sk = data['sk']

    return xk, ydelta, xkdelta, sk


if __name__ == "__main__":
    d = 128
    
    # ========================================================
    # 1. 生成训练集 (Train Set)
    # ========================================================
    # 固定随机种子，确保训练集可重复
    random.seed(42)
    np.random.seed(42)
    print("开始生成训练集 (Train Dataset)...")
    create_dataset(num_samples=100, dataset_path="data/train_dataset.npz")
    
    # ========================================================
    # 2. 生成验证/测试集 (Validation/Test Set)
    # ========================================================
    # 更改随机种子！极其重要！确保测试集的几何图形位置、大小与训练集完全不同
    random.seed(999)
    np.random.seed(999)
    print("开始生成验证/测试集 (Validation Dataset)...")
    create_dataset(num_samples=20, dataset_path="data/val_dataset.npz")
    
    print("所有数据集生成完成!")
