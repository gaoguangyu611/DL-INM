import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import random
from tqdm import tqdm
from data.tikhonov import compute_sk
from data.landweber import landweber_schl

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

    # 参数设置
    d = 128
    theta = np.linspace(0, 135, 60)
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

        # 生成自定义真解图像
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

        step_size = 2e-7
        im_LI, res_history, error_list1, niter = landweber_schl(x_k, gt_1d, R, y_delta, tau, delta, step_size, 2000, early_stop=True)
        print(f"sample{idx+1},step {niter}: error = {error_list1[-1]:.4e},shape = {im_LI.shape}")
            
        xk_all.append(gt_1d)
        ydelta_all.append(y_delta)
        xkdelta_all.append(im_LI.copy())

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


# 主程序 -----------------------------------------------------------------
if __name__ == "__main__":
    d = 128
    random.seed(42)
    np.random.seed(42)

    ground_truths, data, reconstructions = create_dataset(num_samples=100, dataset_path = "data/res_dataset.npz")


    # # 随机选择并绘制几个样本
    # plot_samples(d, ground_truths, reconstructions, num_samples=3)

    print("数据集生成完成!")
    print(f"真解图像 xk 形状: {ground_truths.shape}")
    print(f"投影数据 ydelta 形状: {data.shape}")
    print(f"重建图像 xkdelta 形状: {reconstructions.shape}")

