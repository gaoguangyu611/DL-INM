import numpy as np
import sys


def cg_inexact_newton(F_prime, F_prime_adj, b_nl, alpha, mu_k, cg_max_iter=100, cg_tol=1e-6):
    """
    带 Inexact Newton 截断的内部 CG 求解器
    求解 (F'*F' + alpha*I)s = F'*b_nl

    参数:
    F_prime: 前向 Fréchet 导数算子 F'(x_k)
    F_prime_adj: 伴随算子 F'(x_k)^*
    b_nl: 非线性残差 y_delta - F(x_k)
    alpha: 当前的 Tikhonov 正则化参数
    mu_k: 不精确牛顿条件的系数

    返回:
    s: 求解出的步长
    iters: 实际 CG 迭代步数
    met_condition: 是否成功满足 Inexact Newton 条件
    """
    bnl_2d = b_nl.reshape(60, 128)
    norm_b_nl = max([np.linalg.norm(bnl_2d[i, :]) for i in range(60)])
    # norm_b_nl = np.linalg.norm(b_nl)
    target_res = mu_k * norm_b_nl  # 内部停止的绝对参考标准

    # 初始化
    s = np.zeros_like(F_prime_adj(b_nl))  # 步长初值 s_0 = 0
    w = np.zeros_like(b_nl)               # 维护 w = F_prime(s)，初值为 0

    # 构造右端项 b_cg = F'* b_nl
    b_cg = F_prime_adj(b_nl)
    r_cg = b_cg.copy()                    # 初始正规残差
    p = r_cg.copy()                       # 初始共轭方向

    rsold = np.sum(r_cg * r_cg)
    r_lin = w - b_nl                      # 线性化残差: F_prime(s) - b_nl
    r_lin_2d = r_lin.reshape(60, 128)
    norm_r_lin = max([np.linalg.norm(r_lin_2d[i, :]) for i in range(60)])

    # 如果初始状态就已经满足 (比如 b_nl 很小)
    if norm_r_lin <= target_res:
        return s, 0, True

    for i in range(cg_max_iter):
        # 1. 计算 v = F_prime(p)
        v = F_prime(p)

        # 2. 计算 Ap = F_prime_adj(v) + alpha * p
        Ap = F_prime_adj(v) + alpha * p

        # 3. CG 步长
        p_Ap = np.sum(p * Ap)
        # 防止除以 0 的极小概率事件
        if p_Ap < 1e-20:
            break
        alpha_cg = rsold / p_Ap

        # 4. 更新步长 s 和 正规残差 r_cg
        s = s + alpha_cg * p
        r_cg = r_cg - alpha_cg * Ap

        # 5. 免费更新 w = F_prime(s_new) !!
        w = w + alpha_cg * v

        # --- 检查 Inexact Newton 截断条件 ---
        r_lin = w - b_nl                      # 线性化残差: F_prime(s) - b_nl
        r_lin_2d = r_lin.reshape(60, 128)
        norm_r_lin = max([np.linalg.norm(r_lin_2d[i, :]) for i in range(60)])

        if norm_r_lin <= target_res:
            # 成功满足 Inexact Newton 条件，提前截断 CG！
            return s, i + 1, True

        # --- 标准的 CG 跌代延续 ---
        rsnew = np.sum(r_cg * r_cg)

        # 理论保障：如果 alpha 使得 Inexact Newton 条件无法达到，
        # 则当正规方程残差足够小时(求解精确解了)，强制跳出防止死循环
        if np.sqrt(rsnew) < cg_tol:
            return s, i + 1, False

        p = r_cg + (rsnew / rsold) * p
        rsold = rsnew


    # 如果到达这里，说明CG已经完成最大迭代次数但未满足条件
    print(f"[错误] CG方法未能在 {cg_max_iter} 次迭代内达到要求。线性残差: {norm_r_lin:.4e}, 目标: {target_res:.4e}")
    print("程序终止。")
    sys.exit(1)  # 终止程序



def compute_sk(xk, R, y, alpha_start, q, mu_k, max_alpha_trials=20):
    """
    求解 (F'*F' + alpha*I) s_k = F'*(y - F(x))
    并通过衰减 alpha 确保满足 Inexact Newton 条件: ||F's_k - b_nl|| <= mu_k * ||b_nl||

    参数:
    xk: 当前迭代点
    R: 线性算子
    y: 观测数据 y_delta
    alpha_start: 当前尝试的初始 alpha
    q: alpha 的衰减系数 (例如 0.5)
    mu_k: Inexact Newton 容差系数 (例如 0.9)
    max_alpha_trials: 防止死循环的最大衰减次数

    返回:
    sk: 满足条件的步长
    alpha: 成功满足条件时所使用的 alpha (可作为下一步的参考)
    """

    # 1. 计算当前的非线性残差 b_nl
    Rxk = R @ xk
    Fxk = Rxk ** 2
    b_nl = y - Fxk
    bnl_2d = b_nl.reshape(60, 128)
    norm_b_nl = max([np.linalg.norm(bnl_2d[i, :]) for i in range(60)])
    # norm_b_nl = np.linalg.norm(b_nl)
    target_res = mu_k * norm_b_nl

    # 2. 局部算子封装
    def F_prime(s):
        return 2 * Rxk * (R @ s)

    def F_prime_adj(v):
        return 2 * R.T @ (Rxk * v)

    alpha = alpha_start

    # 3. 动态寻找 alpha 循环
    for trial in range(max_alpha_trials):

        # 调用内层 CG
        sk, cg_iters, met_condition = cg_inexact_newton(
            F_prime, F_prime_adj, b_nl, alpha, mu_k
        )

        # 为确保万无一失，显式验证一下实际的线性化残差
        res_2d = (F_prime(sk) - b_nl).reshape(60, 128)
        lin_res = max([np.linalg.norm(res_2d[i, :]) for i in range(60)])
        # lin_res = np.linalg.norm(F_prime(sk) - b_nl)

        if met_condition or lin_res <= target_res:
            print(f"    [alpha寻找成功] trial={trial}, alpha={alpha:.2e}, "
                  f"CG步数={cg_iters}, 线性残差={lin_res:.4e} <= {target_res:.4e}")
            return sk, alpha
        else:
            print(f"    [alpha过大] trial={trial}, alpha={alpha:.2e}, "
                  f"线性残差={lin_res:.4e} > {target_res:.4e}. 进行衰减(x{q})...")
            # 衰减 alpha，减弱正则化惩罚，使得下一次的 s_k 残差更小
            alpha *= q

    print("[警告] 达到最大 alpha 衰减次数，仍未满足 Inexact Newton 条件，返回当前近似步长。")
    return sk, alpha
