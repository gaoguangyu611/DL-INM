import numpy as np


def landweber_schl(u0, gt_np, R, ydelta, tau, delta, step_size, nitermax, early_stop=True):
    """
    求解非线性算子方程 (Rx)^2 = y_delta 的带噪声 Landweber 迭代重建算法

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
    ydelta_2d = ydelta.reshape(60, 128)

    # 初始投影与残差计算
    raduk = R @ uk
    radukq = raduk ** 2
    Fxk_2d = radukq.reshape(60, 128)
    diffsinog = radukq - ydelta

    res_history = []  # 用于记录残差历史
    error_list = []
    error = np.linalg.norm(uk - gt_np) / np.linalg.norm(gt_np)
    error_list.append(error)

    # 主迭代循环
    for niter in range(nitermax):

        # 1. 计算当前残差范数
        diffnorm = max([np.linalg.norm(ydelta_2d[i, :] - Fxk_2d[i, :]) for i in range(60)])

        res_history.append(diffnorm)

        # 2. 检查 Morozov 偏差原理停止准则
        if early_stop and diffnorm < tau * delta:
            print(f"满足偏差原理提前终止，迭代次数: {niter}, 最终残差: {diffnorm:.4e}")
            break

        # 3. 计算 Fréchet 导数的伴随 F'(u_k)^* (F(u_k) - y^delta)
        Rxy = raduk * diffsinog  # (Ru_k) \odot ((Ru_k)^2 - y^delta)
        RTuk = R.T @ Rxy
        RTuk *= 2                # 2 * R^T[ (Ru_k) \odot ((Ru_k)^2 - y^delta) ]

        # 4. Landweber 更新
        uk = uk - step_size * RTuk
        error = np.linalg.norm(uk - gt_np) / np.linalg.norm(gt_np)
        error_list.append(error)

        # 5. 更新下一次迭代所需的投影和残差
        raduk = R @ uk
        radukq = raduk ** 2
        diffsinog = radukq - ydelta
        Fxk_2d = radukq.reshape(60, 128)

    else:
        # 如果跑满了 nitermax 都没有 break
        if early_stop:
            print(f"警告: 达到最大迭代步数 {nitermax}，未满足偏差原理。最终残差: {diffnorm:.4e}")

    return uk, res_history, error_list, niter
