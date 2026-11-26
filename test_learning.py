import torch
import numpy as np
import os
from inference_models import UNet
from train_residual import UNetres
from train_nett import UNetNETT
from train_inett import UNetinett
from batch_runtime import (compute_sk, begin_method, checkpoint_sample,
    get_operator, get_dataset, get_pinv, learning_batch, warmup, record_dl_iteration, record_partial)
from functools import lru_cache
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import h5py
import time

from parameter_audit import verify_conditions
DL_TUNING = {'L': 0.4, 'Cbeta': 0.01, 'mu': 0.8, 'p': 0.8, 'sigma0': 8.0, 'gamma': 0.875, 'tau': 1.5, 'eta': 1e-06, 'xi': 0.01, 'C_mu': 0.95, 'out_max': 50}
DL_TUNING_AUDIT = verify_conditions(DL_TUNING)
OUTER_LINE_SEARCH_DECAY = 0.125
OUTER_LINE_SEARCH_TRIALS = 7


def select_outer_relaxation(xk, sk_raw, R, y_delta, residual_norm,
                            alpha_initial=0.875):
    """Return a tested, residual-decreasing alpha_k, or None if none exists."""
    for i in range(OUTER_LINE_SEARCH_TRIALS):
        alpha_k = round(alpha_initial - OUTER_LINE_SEARCH_DECAY * i, 10)
        if alpha_k <= 0:
            break
        trial_residual = float(np.linalg.norm(R @ (xk + alpha_k * sk_raw) - y_delta))
        if np.isfinite(trial_residual) and trial_residual < residual_norm:
            return float(alpha_k)
    return None


def projected_network_correction(model, xk, sk_raw, device,
                                 bound_L=0.4, epsilon0=1e-8):
    """Construct U_theta(x,s) using the same per-sample projection as training.

    sk_raw is the unrelaxed Newton direction, NOT alpha_k * sk_raw.
    Returns projected output, projection factor, raw-step norm and net-output norm.
    """
    x_tensor = torch.as_tensor(xk, dtype=torch.float32, device=device).view(1, 1, 128, 128)
    s_tensor = torch.as_tensor(sk_raw, dtype=torch.float32, device=device).view(1, 1, 128, 128)
    raw_output = model(x_tensor, s_tensor)
    norm_s = torch.linalg.vector_norm(s_tensor)
    norm_net = torch.linalg.vector_norm(raw_output)
    projection_scale = torch.clamp(bound_L * norm_s / (norm_net + epsilon0), max=1.0)
    U_theta = projection_scale * raw_output
    return U_theta, float(projection_scale.item()), float(norm_s.item()), float(norm_net.item())


def sync_cuda(device):
    """用于准确统计包含 GPU 运算的方法运行时间。"""
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()

@lru_cache(maxsize=4)
def load_trained_model(model_path, input1_channels=1, input2_channels=1, n_classes=1):
    model = UNet(
        input1_channels=input1_channels,
        input2_channels=input2_channels,
        n_classes=n_classes,
        bilinear=True
    )
    checkpoint = torch.load(model_path, map_location=('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint)
    return model

@lru_cache(maxsize=4)
def load_trained_model_res(model_path, input_channels=1, n_classes=1):
    model = UNetres(
        input_channels=input_channels,
        n_classes=n_classes,
        bilinear=True
    )
    checkpoint = torch.load(model_path, map_location=('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint)
    return model

@lru_cache(maxsize=4)
def load_trained_model_iNETT(model_path, input_channels=1, n_classes=1):
    model = UNetinett(
        in_channels=input_channels,
        out_channels=n_classes,
        bilinear=True
    )
    checkpoint = torch.load(model_path, map_location=('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint)
    return model



def _test_one_sample(model_path, model_path_res, model_path_iNETT, dataset_path, out_max, tau, device='cuda', sample_index=7):
    """
    测试模型性能，专门针对单一样本(OOD混合分布图像)进行测试。
    """
    results = {
        'indices': [sample_index],
        'original_images':[],
        'res_steps': [],
        'res_recons':[],
        'res_error':[],
        'res_residual':[],
        'res_time': [],
        'res_inner_total': [],

        'DL_sk_norm': [],
        'model_item_norm':[],
        'DL_recons':[],
        'DL_steps': [],
        'DL_error':[],
        'beta': [],
        'DL_residual': [],
        'DL_time': [],
        'DL_inner_total': [],
        'DL_scale_factor': [],

        'IN_steps': [],
        'IN_recons': [],
        'IN_error': [],
        'IN_residual': [],
        'IN_time': [],                  # 新增
        'IN_inner_total': [],           # 新增

        'dc_steps': [],
        'dc_recons': [],
        'dc_error': [],
        'dc_residual': [],
        'dc_time': [],                  # 新增
        'dc_inner_total': [],           # 新增

        'NETT_recons': [],
        'NETT_steps': [],
        'NETT_error': [],
        'NETT_residual': [],

        'iNETT_recons': [],
        'iNETT_steps': [],
        'iNETT_error': [],
        'iNETT_residual': [],
        'iNETT_time': [],               # 新增
        'iNETT_inner_total': []         # 新增
    }
    
    print(f"Loading custom OOD test sample from {dataset_path}...")

    # ========================================================
    # 1. 加载真解图像与物理算子
    # ========================================================
    test_dataset = get_dataset(dataset_path)
    ground_truth= test_dataset['xk'][sample_index].flatten()
    y_delta = test_dataset['ydelta'][sample_index].flatten()
    print(f"ydelata shape: {y_delta.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")
    gt_np = ground_truth.flatten()

    # 加载物理算子 R (7680, 16384)
    R = get_operator()

    # =========================================================
    # 模型加载：属于离线准备，不计入 reconstruction runtime
    # =========================================================

    modelres = load_trained_model_res(model_path_res)
    modelres = modelres.to(device)
    modelres.eval()

    model1 = load_trained_model(model_path)
    model1 = model1.to(device)
    model1.eval()

    model_inett = load_trained_model_iNETT(model_path_iNETT)
    model_inett = model_inett.to(device)
    model_inett.eval()

    warmup(model1,device,two_inputs=True)
    warmup(modelres,device)
    warmup(model_inett,device)

    for p in model_inett.parameters():
        p.requires_grad_(False)
    
    # ========================================================
    # 2. 模拟物理观测生成 y_delta
    # ========================================================
    
    # # 无噪声的精确投影 (R@x)^2
    y_exact = R @ gt_np

    # 计算噪声水平
    delta = np.linalg.norm(y_delta-y_exact)
    stop_threshold = tau * delta
    # ========================================================

    results['original_images'].append(gt_np)
    print(f"Sample OOD - Noise Level (delta): {delta:.4e}, Target Stop: {stop_threshold:.4e}")

    with torch.no_grad():
        for example in ['res', 'DL_INM', 'iNETT']:
            begin_method(example)
            sync_cuda(device)
            method_start_time = time.perf_counter()
            
            # 初始解必须是 1D 向量 (16384,)
            xk = np.ones((128*128,)) * 0.01
            Rxk = R @ xk
            residual_norm = np.linalg.norm(Rxk - y_delta)
            print(f"\n--- Running method: {example} ---")
            print(f"step {0}: Residual Norm = {residual_norm:.4e}")
            
            k = 0
            error_list = []
            error_list2 = [] 
            sk_norm_list =[]
            model_norm_list =[]
            beta_list = []
            residual_list = []
            residual_list2 = []

            inner_iter_total = 0
            scale_factor_list = []
            

            if example == 'res':

                reserror = np.linalg.norm(xk - gt_np) / np.linalg.norm(gt_np)
                error_list.append(reserror)
                error_list2.append(reserror)
                residual_list.append(residual_norm)
                residual_list2.append(residual_norm)
                
                while k < out_max:
                    k += 1
                    sk, alpha, inner_count = compute_sk(xk, R, y_delta, alpha_start=8, q=0.8, mu_k=0.8)
                    inner_iter_total += inner_count
                    # ---------------- 加入回溯线搜索 (Line Search) ----------------
                    gamma = 0.875
                    for _ in range(6): # 最多回溯 10 次
                        xk_trial = xk + gamma * sk
                        Rxk1 = R @ xk_trial
                        res_trial = np.linalg.norm(Rxk1 - y_delta)
                        if res_trial < residual_norm:   
                            break
                        gamma += -0.125
                    # ------------------------------------------------------------------------
                    
                    xk = xk + gamma * sk
                    Rxk = R @ xk
                    residual_norm = np.linalg.norm(Rxk - y_delta)
                    print(f"IN_step {k}: Residual Norm = {residual_norm:.4e}, Target = {stop_threshold:.4e}")
                    
                    reserror = np.linalg.norm(xk - gt_np) / np.linalg.norm(gt_np)
                    error_list.append(reserror)
                    error_list2.append(reserror)
                    residual_list.append(residual_norm)
                    residual_list2.append(residual_norm)

                    if residual_norm <= stop_threshold:
                        break
                sync_cuda(device)
                IN_elapsed_time = time.perf_counter() - method_start_time
                
                results['IN_recons'].append(xk)
                results['IN_steps'].append(k)
                results['IN_error'].append(error_list2)
                results['IN_residual'].append(residual_list2)
                results['IN_time'].append(IN_elapsed_time)
                results['IN_inner_total'].append(inner_iter_total)

                print(
                    f"[IN] outer={k}, "
                    f"inner_total={inner_iter_total}, "
                    f"time={IN_elapsed_time:.4f} s"
                )
                
                xk_tensor = torch.tensor(xk, dtype=torch.float32, device=device).view(1, 1, 128, 128)
                model_output = modelres(xk_tensor)
                model_output_np = model_output.detach().cpu().numpy().flatten()
                xk1 = xk + 0.65*model_output_np
                Rxk1 = R @ xk1
                residual_norm = np.linalg.norm(Rxk1 - y_delta)
                reserror = np.linalg.norm(xk1 - gt_np) / np.linalg.norm(gt_np)
                error_list.append(reserror)
                residual_list.append(residual_norm)
                print(f"Residual Norm = {residual_norm:.4e}, error: {reserror:.4e}")

                sync_cuda(device)
                RES_elapsed_time = time.perf_counter() - method_start_time

                results['res_recons'].append(xk1)
                results['res_steps'].append(k+1)
                results['res_error'].append(error_list.copy())
                results['res_residual'].append(residual_list.copy())
                results['res_time'].append(RES_elapsed_time)
                results['res_inner_total'].append(inner_iter_total)
                print(
                    f"[IN-RES] "
                    f"IN_outer={k}, "
                    f"inner_total={inner_iter_total}, "
                    f"time={RES_elapsed_time:.4f} s"
                )

                R_pinv = get_pinv()
                xk_dc = xk1 - R_pinv @ (R @ xk1 - R @ xk)
                Rxk2 = R @ xk_dc
                dc_residual_norm = np.linalg.norm(Rxk2 - y_delta)
                dc_error = np.linalg.norm(xk_dc - gt_np) / np.linalg.norm(gt_np)
                error_list.append(dc_error)
                residual_list.append(dc_residual_norm)
                print(f"dc_Residual Norm = {dc_residual_norm:.4e}, dc_error = {dc_error:.4e}")
                sync_cuda(device)
                DC_elapsed_time = time.perf_counter() - method_start_time
                results['dc_recons'].append(xk_dc)
                results['dc_steps'].append(k+2)
                results['dc_error'].append(error_list)
                results['dc_residual'].append(residual_list)
                results['dc_time'].append(DC_elapsed_time)
                results['dc_inner_total'].append(inner_iter_total)

                print(
                    f"[IN-DC] "
                    f"IN_outer={k}, "
                    f"inner_total={inner_iter_total}, "
                    f"time={DC_elapsed_time:.4f} s"
                )


                

            if example == 'DL_INM':
                error_res = np.linalg.norm(xk - gt_np) / np.linalg.norm(gt_np)
                error_list.append(error_res)
                residual_list.append(residual_norm)
                alpha_history = []
                stopped_by_discrepancy = residual_norm <= stop_threshold
                line_search_failed = False

                while k < out_max and residual_norm > stop_threshold:
                    sk_raw, _inner_parameter, inner_count = compute_sk(
                        xk, R, y_delta,
                        alpha_start=DL_TUNING['sigma0'],
                        q=DL_TUNING['p'],
                        mu_k=DL_TUNING['mu'],
                    )
                    sk_raw = np.asarray(sk_raw, dtype=np.float64).reshape(-1)
                    if sk_raw.size != xk.size or not np.all(np.isfinite(sk_raw)):
                        raise ValueError("compute_sk returned an invalid raw Newton step")
                    inner_iter_total += inner_count

                    alpha_k = select_outer_relaxation(
                        xk=xk, sk_raw=sk_raw, R=R, y_delta=y_delta,
                        residual_norm=residual_norm,
                        alpha_initial=DL_TUNING['gamma'],
                    )
                    if alpha_k is None:
                        print(f"[DL-INM] Line search failed at outer iteration {k}; "
                              "stopping without applying an untested step.")
                        line_search_failed = True
                        break

                    psi_necessary = (alpha_k + DL_TUNING['L']) ** 2 - 2.0 * alpha_k
                    if psi_necessary >= 0:
                        raise ValueError(
                            f"alpha_k={alpha_k:.3f} cannot satisfy paper Eq. (3.9) "
                            f"even before the positive rho/C_mu term is added. "
                            "Align the admissible line-search range in BOTH "
                            "data generation and testing."
                        )

                    U_theta, projection_scale, norm_sk, norm_unet = projected_network_correction(
                        model1, xk, sk_raw, device,
                        bound_L=DL_TUNING['L'], epsilon0=1e-8,
                    )
                    projected_norm = float(torch.linalg.vector_norm(U_theta).item())
                    U_np = U_theta.detach().cpu().numpy().reshape(-1)
                    if projected_norm > DL_TUNING['L'] * norm_sk + 1e-5:
                        raise AssertionError("U_theta violates the theoretical norm bound")

                    # Residual from x_k BEFORE the update (Eq. 2.3).
                    beta_k = min(1.0, DL_TUNING['Cbeta'] * residual_norm)
                    beta_list.append(beta_k)
                    scale_factor_list.append(projection_scale)
                    alpha_history.append(alpha_k)
                    sk_norm_list.append(norm_sk)
                    model_norm_list.append(projected_norm)

                    # DL-INM outer update (paper Algorithm 1).
                    xk = xk + alpha_k * sk_raw + beta_k * U_np
                    residual_norm = float(np.linalg.norm(R @ xk - y_delta))
                    k += 1
                    error_res = np.linalg.norm(xk - gt_np) / np.linalg.norm(gt_np)
                    error_list.append(error_res)
                    residual_list.append(residual_norm)
                    record_dl_iteration(k, alpha_k, beta_k, projection_scale, norm_sk,
                                        norm_unet, projected_norm, residual_norm, error_res)

                    print(
                        f"DL-INM_step {k}: "
                        f"alpha_k={alpha_k:.4f}, "
                        f"beta_k={beta_k:.4f}, "
                        f"projection_scale={projection_scale:.4f}, "
                        f"inner={inner_count}, inner_total={inner_iter_total}, "
                        f"||s_raw||={norm_sk:.4e}, "
                        f"||N_theta||={norm_unet:.4e}, "
                        f"||U_theta||={projected_norm:.4e}, "
                        f"residual={residual_norm:.4e}, RE={error_res:.4f}"
                    )

                stopped_by_discrepancy = residual_norm <= stop_threshold
                sync_cuda(device)
                DL_elapsed_time = time.perf_counter() - method_start_time

                results['DL_recons'].append(xk)
                results['DL_steps'].append(k)
                results['DL_error'].append(error_list)
                results['DL_sk_norm'].append(sk_norm_list)
                results['beta'].append(beta_list)
                results['model_item_norm'].append(model_norm_list)
                results['DL_residual'].append(residual_list)
                results['DL_time'].append(DL_elapsed_time)
                results['DL_inner_total'].append(inner_iter_total)
                results['DL_scale_factor'].append(scale_factor_list)
                print(
                    f"[DL-INM] outer={k}, inner_total={inner_iter_total}, "
                    f"time={DL_elapsed_time:.4f}s, RE={error_res:.4f}, "
                    f"discrepancy_met={stopped_by_discrepancy}, "
                    f"line_search_failed={line_search_failed}"
                )
                print("DL-INM alpha_k (Newton relaxation):",
                      [f"{a:.4f}" for a in alpha_history])
                print("DL-INM projection scales:",
                      [f"{p:.4f}" for p in scale_factor_list])


            if example == 'iNETT':

                image_size = 128
                N = image_size * image_size
                R_tensor = torch.tensor(R, dtype=torch.float32, device=device)
                y_delta_tensor = torch.tensor(y_delta, dtype=torch.float32, device=device).view(-1, 1)

                # M 是观测维度，用于归一化 data fidelity，避免数值过大
                M = y_delta_tensor.numel()

                # 初始点
                xk_outer = xk.copy().reshape(-1, 1)

                # 初始误差记录
                residual_norm = np.linalg.norm(R @ xk_outer - y_delta.reshape(-1, 1))
                reserror = np.linalg.norm(xk_outer.reshape(-1) - gt_np.reshape(-1)) / np.linalg.norm(gt_np.reshape(-1))

                error_list.append(reserror)
                residual_list.append(residual_norm)

                def forward_torch(x_vec: torch.Tensor) -> torch.Tensor:
                    return torch.matmul(R_tensor, x_vec)

                def data_fidelity(x_vec: torch.Tensor) -> torch.Tensor:
                    """
                    Normalized data fidelity:
                        D(x) = 1/(2M) ||R x - y_delta||_2^2
                    """
                    residual = forward_torch(x_vec) - y_delta_tensor
                    return 0.5 / M * torch.sum(residual ** 2)

                def regularizer_R(x_vec: torch.Tensor) -> torch.Tensor:
                    """
                    UCNN regularizer:
                        R_net(x) = a ||x||_2^2 + ||Phi_c(x)||_2^2
                    """
                    x_img = x_vec.view(1, 1, 128, 128)

                    # 模型已经有 regularizer 方法
                    R_value = model_inett.regularizer(x_img)

                    # model.regularizer(x_img) 通常返回 [B]，这里 B=1
                    return R_value / N
                @torch.enable_grad()
                def grad_regularizer_R(x_vec: torch.Tensor) -> torch.Tensor:
                    """
                    Compute xi = grad R(x).
                    """
                    x_var = x_vec.detach().clone().requires_grad_(True)
                    R_value = regularizer_R(x_var)
                    grad_R = torch.autograd.grad(R_value, x_var, create_graph=False)[0]
                    return grad_R.detach()

                @torch.enable_grad()
                def grad_data_fidelity(x_vec: torch.Tensor) -> torch.Tensor:
                    """
                    Compute gradient of normalized data fidelity:
                        grad D(x) = 1/M * R^T (R x - y_delta)
                    """
                    x_var = x_vec.detach().clone().requires_grad_(True)
                    D_value = data_fidelity(x_var)
                    grad_D = torch.autograd.grad(D_value, x_var, create_graph=False)[0]
                    return grad_D.detach()
                
                def project_image_(x_vec: torch.Tensor) -> None:
                    """
                    Projection onto the valid image range.

                    For normalized Shepp-Logan type phantoms, [0, 1] is reasonable.
                    If you do not want the upper bound, remove max=1.0.
                    """
                    with torch.no_grad():
                        x_vec.clamp_(min=0.0, max=1.0)

                @torch.no_grad()
                def evaluate_subproblem_value(
                    x_vec: torch.Tensor,
                    x_anchor: torch.Tensor,
                    xi_anchor: torch.Tensor,
                    R_anchor: torch.Tensor,
                    alpha_n: float,
                ) -> float:
                    """
                    Evaluate the current Bregman subproblem value without building graph.
                    """
                    D_val = data_fidelity(x_vec)
                    R_val = regularizer_R(x_vec)
                    B_val = R_val - R_anchor - torch.sum(xi_anchor * (x_vec - x_anchor))
                    total = D_val + alpha_n * B_val
                    return total.item()

                x_prev = torch.tensor(xk_outer, dtype=torch.float32, device=device, requires_grad=False)

                # 因为 R_net 是可微的，所以 xi_0 = grad R(x_0)
                xi_prev = grad_regularizer_R(x_prev)

                alpha0 = 1e4
                alpha_decay = 0.85

                inner_steps = 300
                lr_inner = 3e-4
                outer_damping = 0.5
                bregman_negative_tol = -1e-4

                iNETT_inner_total = 0

                while k < out_max:
                    k += 1

                    alpha_n = alpha0 * (alpha_decay ** (k - 1))

                    x_anchor = x_prev.detach()
                    xi_anchor = xi_prev.detach()
                    R_anchor = regularizer_R(x_anchor).detach()

                    x_inner = x_anchor.clone().detach().requires_grad_(True)

                    optimizer = torch.optim.Adam([x_inner], lr=lr_inner)

                    best_x_inner = x_inner.detach().clone()
                    best_inner_obj = float("inf")
                    best_rel_error = float("inf")
                    best_x_inett = None
                    best_iter = 0

                    last_loss_data = None
                    last_bregman = None
                    last_loss = None

                    
                    with torch.enable_grad():
                        for inner_iter in range(inner_steps):

                            iNETT_inner_total += 1

                            optimizer.zero_grad()

                            loss_data = data_fidelity(x_inner)

                            R_inner = regularizer_R(x_inner)
                            bregman = R_inner - R_anchor - torch.sum(xi_anchor * (x_inner - x_anchor))

                            loss = loss_data + alpha_n * bregman

                            optimizer.zero_grad()
                            loss.backward()

                            # 对 x_inner 的梯度裁剪，防止局部震荡
                            torch.nn.utils.clip_grad_norm_([x_inner], max_norm=1.0)

                            optimizer.step()

                            # 关键：投影到合理图像范围，否则容易产生网格噪声
                            project_image_(x_inner)

                            # 保存当前内层最优点，避免 Adam 最后几步震荡
                            with torch.no_grad():
                                current_obj = evaluate_subproblem_value(
                                    x_inner,
                                    x_anchor,
                                    xi_anchor,
                                    R_anchor,
                                    alpha_n,
                                )

                                if np.isfinite(current_obj) and current_obj < best_inner_obj:
                                    best_inner_obj = current_obj
                                    best_x_inner = x_inner.detach().clone()

                            last_loss_data = loss_data.detach()
                            last_bregman = bregman.detach()
                            last_loss = loss.detach()

                    # 使用内层最优点，而不是 Adam 最后一步
                    x_candidate = best_x_inner.detach()

                    # 可选外层阻尼，防止外层跳太远
                    if outer_damping < 1.0:
                        x_new = (1.0 - outer_damping) * x_anchor + outer_damping * x_candidate
                    else:
                        x_new = x_candidate

                    project_image_(x_new)

                    grad_D_new = grad_data_fidelity(x_new)
                    xi_new = xi_prev - (1.0 / alpha_n) * grad_D_new
                    xk_outer = x_new.detach().cpu().numpy().reshape(-1, 1)

                    Rxk_np = R @ xk_outer
                    residual_norm = np.linalg.norm(Rxk_np - y_delta.reshape(-1, 1))

                    error_res = np.linalg.norm(xk_outer.reshape(-1) - gt_np.reshape(-1)) / np.linalg.norm(gt_np.reshape(-1))

                    error_list.append(error_res)
                    residual_list.append(residual_norm)
                    if error_res < best_rel_error:
                        best_rel_error = error_res
                        best_x_inett = xk_outer.copy()
                        best_iter = k

                    x_min = float(x_new.min().detach().cpu())
                    x_max = float(x_new.max().detach().cpu())
                    x_mean = float(x_new.mean().detach().cpu())

                    bregman_value = last_bregman.item() if last_bregman is not None else float("nan")
                    data_value = last_loss_data.item() if last_loss_data is not None else float("nan")
                    loss_value = last_loss.item() if last_loss is not None else float("nan")

                    print(
                        f"iNETT outer {k:03d}: "
                        f"alpha={alpha_n:.3e}, "
                        f"residual={residual_norm:.4e}, "
                        f"rel_error={error_res:.4e}, "
                        f"data={data_value:.4e}, "
                        f"bregman={bregman_value:.4e}, "
                        f"loss={loss_value:.4e}, "
                        f"x_range=[{x_min:.3f}, {x_max:.3f}], "
                        f"x_mean={x_mean:.3f}"
                    )

                    if bregman_value < bregman_negative_tol:
                        print(
                            f"[Warning] Bregman distance is negative: {bregman_value:.4e}. "
                            f"This may indicate insufficient convexity, numerical instability, "
                            f"or an overly aggressive inner optimizer."
                        )
                    if "stop_threshold" in locals():
                        if residual_norm <= stop_threshold:
                            print(
                                f"iNETT stopped by discrepancy principle: "
                                f"residual={residual_norm:.4e}, "
                                f"threshold={stop_threshold:.4e}"
                            )
                            break
                    x_prev = x_new.detach()
                    xi_prev = xi_new.detach()
                xk_outer = best_x_inett
                print(f"Best iNETT iter: {best_iter}, best rel_error: {best_rel_error:.4e}")

                sync_cuda(device)
                iNETT_elapsed_time = time.perf_counter() - method_start_time
                results['iNETT_recons'].append(xk_outer)
                results['iNETT_steps'].append(k)
                results['iNETT_error'].append(error_list)
                results['iNETT_residual'].append(residual_list)
                results['iNETT_time'].append(iNETT_elapsed_time)
                results['iNETT_inner_total'].append(iNETT_inner_total)

                print(
                    f"[iNETT] "
                    f"outer={k}, "
                    f"inner_total={iNETT_inner_total}, "
                    f"time={iNETT_elapsed_time:.4f} s"
                )
            record_partial(results, sample_index, example)

        print(f"Test Done. Final Error - IN: {results['IN_error'][-1][-1]:.4f}, res: {results['res_error'][-1][-1]:.4f}, DL-INM: {results['DL_error'][-1][-1]:.4f}")
        print("-" * 50)
        print("\n" + "=" * 80)
        print("COMPUTATIONAL COST SUMMARY")
        print("=" * 80)

        print(
            f"IN       : "
            f"outer={results['IN_steps'][-1]}, "
            f"inner={results['IN_inner_total'][-1]}, "
            f"time={results['IN_time'][-1]:.4f} s"
        )

        print(
            f"IN-RES   : "
            f"inner={results['res_inner_total'][-1]}, "
            f"time={results['res_time'][-1]:.4f} s"
        )

        print(
            f"IN-DC    : "
            f"inner={results['dc_inner_total'][-1]}, "
            f"time={results['dc_time'][-1]:.4f} s"
        )

        print(
            f"DL-INM   : "
            f"outer={results['DL_steps'][-1]}, "
            f"inner={results['DL_inner_total'][-1]}, "
            f"time={results['DL_time'][-1]:.4f} s"
        )

        print(
            f"iNETT    : "
            f"outer={results['iNETT_steps'][-1]}, "
            f"inner={results['iNETT_inner_total'][-1]}, "
            f"time={results['iNETT_time'][-1]:.4f} s"
        )

        print(
            "DL-INM scale_factor:",
            [f"{x:.4f}" for x in results['DL_scale_factor'][-1]]
        )

        print("=" * 80)
            
    return results


def test_model_performance(model_path, model_path_res, model_path_iNETT, dataset_path, out_max, tau, device='cuda', num_samples=30, selected_indices=None):
    return learning_batch(_test_one_sample,model_path,model_path_res,model_path_iNETT,
        dataset_path,out_max,tau,device,num_samples,selected_indices)

def _evaluate_one_sample_and_plot(results, save_dir='test_results'):
    os.makedirs(save_dir, exist_ok=True)
    
    gt = results['original_images'][0].reshape(128, 128)
    resrecon = results['res_recons'][0].reshape(128, 128)
    recon_DL = results['DL_recons'][0].reshape(128, 128)
    recon_in = results['IN_recons'][0].reshape(128, 128)
    recon_dc = results['dc_recons'][0].reshape(128, 128)
    recon_inett = results['iNETT_recons'][0].reshape(128, 128)
    
    data_range = gt.max() - gt.min()
    if data_range == 0: data_range = 1.0
    
    psnr_res, ssim_res = psnr(gt, resrecon, data_range=data_range), ssim(gt, resrecon, data_range=data_range)
    psnr_DL, ssim_DL = psnr(gt, recon_DL, data_range=data_range), ssim(gt, recon_DL, data_range=data_range)
    psnr_in, ssim_in = psnr(gt, recon_in, data_range=data_range), ssim(gt, recon_in, data_range=data_range)
    psnr_dc, ssim_dc = psnr(gt, recon_dc, data_range=data_range), ssim(gt, recon_dc, data_range=data_range)
    psnr_inett, ssim_inett = psnr(gt, recon_inett, data_range=data_range), ssim(gt, recon_inett, data_range=data_range) # 新增

    reserr = results['res_error'][0]
    err_DL = results['DL_error'][0]
    err_IN = results['IN_error'][0]
    
    model_norm = results['model_item_norm'][0]

    residual_IN = results['IN_residual'][0]
    residual_res = results['res_residual'][0]
    residual_DL = results['DL_residual'][0]

    vmin, vmax = gt.min(), gt.max()

    # 图 1：反演结果与绝对误差热力图合集 
    fig, axes = plt.subplots(2, 5, figsize=(27, 10))
    
    im1 = axes[0, 0].imshow(recon_in, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f"IN\nPSNR: {psnr_in:.2f} | SSIM: {ssim_in:.4f}", fontsize=16, fontweight='bold')
    fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im2 = axes[0, 3].imshow(resrecon, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 3].set_title(f"IN-RES\nPSNR: {psnr_res:.2f} | SSIM: {ssim_res:.4f}", fontsize=16, fontweight='bold')
    fig.colorbar(im2, ax=axes[0, 3], fraction=0.046, pad=0.04)

    im3 = axes[0, 2].imshow(recon_dc, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title(f"IN-DC\nPSNR: {psnr_dc:.2f} | SSIM: {ssim_dc:.4f}", fontsize=16, fontweight='bold')
    fig.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    im4 = axes[0, 4].imshow(recon_DL, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 4].set_title(f"DL-INM\nPSNR: {psnr_DL:.2f} | SSIM: {ssim_DL:.4f}", fontsize=16, fontweight='bold')
    fig.colorbar(im4, ax=axes[0, 4], fraction=0.046, pad=0.04)

    im5 = axes[0, 1].imshow(recon_inett, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f"iNETT\nPSNR: {psnr_inett:.2f} | SSIM: {ssim_inett:.4f}", fontsize=16, fontweight='bold')
    fig.colorbar(im5, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    err_map_res = np.abs(gt - resrecon)
    err_map_DL = np.abs(gt - recon_DL)
    err_map_in = np.abs(gt - recon_in)
    err_map_dc = np.abs(gt - recon_dc)
    err_map_iNETT = np.abs(gt - recon_inett)
    evmax = max(err_map_res.max(), err_map_DL.max(), err_map_in.max(), err_map_dc.max(), err_map_iNETT.max())


    im_e1 = axes[1, 0].imshow(err_map_in, cmap='hot', vmin=0, vmax=evmax)
    axes[1, 0].set_title("Error Heatmap (IN)", fontsize=16, fontweight='bold')
    fig.colorbar(im_e1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im_e2 = axes[1, 3].imshow(err_map_res, cmap='hot', vmin=0, vmax=evmax)
    axes[1, 3].set_title("Error Heatmap (IN-RES)", fontsize=16, fontweight='bold')
    fig.colorbar(im_e2, ax=axes[1, 3], fraction=0.046, pad=0.04)

    im_e3 = axes[1, 2].imshow(err_map_dc, cmap='hot', vmin=0, vmax=evmax)
    axes[1, 2].set_title("Error Heatmap (IN-DC)", fontsize=16, fontweight='bold')
    fig.colorbar(im_e3, ax=axes[1, 2], fraction=0.046, pad=0.04)

    im_e4 = axes[1, 4].imshow(err_map_DL, cmap='hot', vmin=0, vmax=evmax)
    axes[1, 4].set_title("Error Heatmap (DL-INM)", fontsize=16, fontweight='bold')
    fig.colorbar(im_e4, ax=axes[1, 4], fraction=0.046, pad=0.04)

    im_e5 = axes[1, 1].imshow(err_map_iNETT, cmap='hot', vmin=0, vmax=evmax)
    axes[1, 1].set_title("Error Heatmap (iNETT)", fontsize=16, fontweight='bold')
    fig.colorbar(im_e5, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ood_test_reconstruction_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图 2：单独的相对误差曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(err_IN)+1), err_IN, 'o-', linewidth=2, color='green', label='IN')
    plt.plot(range(1, len(reserr)+1), reserr, 'x-', linewidth=2, color='red', label='IN-RES')
    plt.plot(range(1, len(err_DL)+1), err_DL, 's-', linewidth=2, color='orange', label='DL-INM')
    plt.xlabel("Iteration Step", fontsize=14)
    plt.ylabel("Relative Error", fontsize=14)
    plt.grid(True, linestyle='--')
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(save_dir, 'ood_test_error_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()


    # 图 3：残差
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(residual_IN)+1), residual_IN, 'o-', linewidth=2, color='green', label='IN')
    plt.plot(range(1, len(residual_res)+1), residual_res, 'x-', linewidth=2, color='red', label='IN-RES')
    plt.plot(range(1, len(residual_DL)+1), residual_DL, 's-', linewidth=2, color='orange', label='DL-INM')
    plt.xlabel("Iteration Step", fontsize=14)
    plt.ylabel("Residual Norm", fontsize=14)
    plt.grid(True, linestyle='--')
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(save_dir, 'ood_test_residual_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # === 图 4：一维截面对比 (1D Profile) - 并列三图横纵互换 ===
    # 创建 1行3列 的画布，尺寸适当拉长以容纳三个子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 需要提取的行索引
    row_indices = [40, 80, 120]
    
    for i, row_idx in enumerate(row_indices):
        ax = axes[i]
        
        # 提取各个方法的切片
        gt_slice = gt[row_idx, :]
        res_slice = resrecon[row_idx, :]
        dc_slice = recon_dc[row_idx, :]
        dl_slice = recon_DL[row_idx, :]
        # nett_slice = recon_nett[row_idx, :]
        inett_slice = recon_inett[row_idx, :]
        
        # 生成像素索引（Y轴的数据）
        pixel_indices = np.arange(len(gt_slice))
        
        # 绘制图表：注意这里将 slice数据 放在X轴位置，pixel_indices 放在Y轴位置
        ax.plot(gt_slice, pixel_indices, 'k--', linewidth=2, label='Ground Truth')
        ax.plot(inett_slice, pixel_indices, 'brown', linewidth=2, linestyle='-.', label='iNETT')
        ax.plot(dc_slice, pixel_indices, 'c-', linewidth=2, label='IN-DC')
        ax.plot(res_slice, pixel_indices, 'r-', linewidth=2, alpha=0.8, label='IN-RES')
        ax.plot(dl_slice, pixel_indices, 'b-', linewidth=2, label='DL-INM')
        # ax.plot(nett_slice, pixel_indices, 'm-', linewidth=1.5, label='NETT')
        
        
        # 设置标题和网格
        ax.set_title(f"Row {row_idx}", fontsize=16, fontweight='bold')
        ax.grid(True, linestyle=':')
        
        # 坐标轴标签互换
        ax.set_xlabel("Intensity", fontsize=14, fontweight='bold')
        if i == 0:
            ax.set_ylabel("Pixel Index", fontsize=14, fontweight='bold') # 只有最左边的图显示Y轴标签

        ax.invert_yaxis()

    
    # 为了避免遮挡曲线，将图例统一放在最后一个子图里（或者你也可以用 bbox_to_anchor 移到图外）
    axes[-1].legend(fontsize=12, loc='best')
    
    # 自动调整子图间距
    plt.tight_layout()
    # tight_layout() 有时会遮挡 suptitle，稍微调整下顶部边距
    plt.subplots_adjust(top=0.88)
    
    # 保存并关闭
    plt.savefig(os.path.join(save_dir, 'ood_test_1d_profile_multi.png'), dpi=300, bbox_inches='tight')
    plt.close()



def evaluate_and_plot(results, save_dir='test_results'):
    """Apply the original plotting function to every requested sample."""
    for position, sample_index in enumerate(results['indices']):
        one = {key: [value[position]] for key, value in results.items()
               if isinstance(value, list) and len(value) > position}
        one['indices'] = [sample_index]
        _evaluate_one_sample_and_plot(one, os.path.join(save_dir, 'sample_%02d' % sample_index))

if __name__ == "__main__":
    MODEL_PATH = "models/final_best.pth"
    MODEL_PATH_RES = "models/res_final_best.pth"  
    MODEL_PATH_NETT = "models/NETT_final_best.pth"     
    MODEL_PATH_iNETT = "models/iNETT_final_best.pth"

    DATASET_PATH = "data/test_dataset.npz"
    SAVE_DIR = "results/learning/plots"

    MAX_ITERATIONS = 50
    TAU = 1.5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"=== Starting OOD Evaluation on {DEVICE.upper()} ===")
    
    if not os.path.exists(MODEL_PATH): raise FileNotFoundError(f"未找到: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH_RES): raise FileNotFoundError(f"未找到: {MODEL_PATH_RES}")
    if not os.path.exists(DATASET_PATH): raise FileNotFoundError(f"未找到: {DATASET_PATH}")
    if not os.path.exists('R_limited.npy'): raise FileNotFoundError("未找到 R_limited.npy")
    results = test_model_performance(
        model_path=MODEL_PATH, 
        model_path_res=MODEL_PATH_RES,
        model_path_iNETT=MODEL_PATH_iNETT,
        dataset_path=DATASET_PATH, 
        out_max=MAX_ITERATIONS, 
        tau=TAU, 
        device=DEVICE,
        num_samples=30,
        selected_indices=list(range(30))
    )

    
    print("\nGenerating evaluation plots...")
    evaluate_and_plot(results, save_dir=SAVE_DIR)
    
    print(f"\nAll Done! 你的高对比度实验结果已保存在 '{SAVE_DIR}' 文件夹中。")