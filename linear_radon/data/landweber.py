import numpy as np


def landweber_radon(u0, gt_np, R, ydelta, tau, delta, step_size, nitermax, early_stop=True):
    """
    求解非线性算子方程 Rx = y_delta 的带噪声 Landweber 迭代重建算法

    参数:
    u0: 初始场 (一维 numpy array)
    R: 拉东算子矩阵 (2D numpy array)
    ydelta: 投影观测数据 (带噪声的数据)
    tau: Morozov 偏差原理的阈值系数 (理论上要求 tau > 1)
    delta: 噪声水平 (即 ||y_delta - y_exact||_2)
    step_size: 迭代步长
    nitermax: 最大迭代次数
    early_stop: 是否使用 Morozov 偏差原理提前终止 (默认 True)

    返回:
    uk: 重建的解
    res_history: 每次迭代的残差范数列表 (方便数值实验画图)
    """
    # 修正缩进
    uk = u0.copy()

    # 初始投影与残差计算
    raduk = R @ uk
    diffsinog = raduk - ydelta

    res_history = []  # 用于记录残差历史
    error_list = []
    error = np.linalg.norm(uk - gt_np) / np.linalg.norm(gt_np)
    error_list.append(error)

    # 主迭代循环
    for niter in range(nitermax):

        # 1. 计算当前残差范数
        diffnorm = np.linalg.norm(diffsinog)
        # print(f"停止目标残差：{tau * delta:.4e}")
        # print(f"迭代步数：{niter:.4e},残差范数: {diffnorm:.4e}")
        res_history.append(diffnorm)

        # 2. 检查 Morozov 偏差原理停止准则
        # 如果你做实验时想强行跑满 nitermax 步，在调用时设 early_stop=False 即可
        if early_stop and diffnorm < tau * delta:
            print(f"满足偏差原理提前终止，迭代次数: {niter}, 最终残差: {diffnorm:.4e}")
            break

        RTuk = R.T @ diffsinog

        # 4. Landweber 更新
        uk = uk - step_size * RTuk
        error = np.linalg.norm(uk - gt_np) / np.linalg.norm(gt_np)
        error_list.append(error)

        # 5. 更新下一次迭代所需的投影和残差
        raduk = R @ uk
        diffsinog = raduk - ydelta

    else:
        # 如果跑满了 nitermax 都没有 break
        if early_stop:
            print(f"警告: 达到最大迭代步数 {nitermax}，未满足偏差原理。最终残差: {diffnorm:.4e}")

    return uk, res_history, error_list, niter
