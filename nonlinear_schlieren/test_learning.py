import torch
import numpy as np
import os
from train_dl_inm import UNet
from train_residual import UNet2
from train_nett import UNetNETT
from runtime import (compute_sk, get_operator, get_dataset, get_pinv,
    get_torch_operator, warmup, begin_method, record_dl_step, finish_traditional, finish_learning,
    mark_learning_stage)
from functools import lru_cache
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

TARGET_METHODS = ["res", "DL_INM", "NETT"]
NETT_MAX = 200

@lru_cache(maxsize=2)
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

@lru_cache(maxsize=2)
def load_trained_model2(model_path, input_channels=1, n_classes=1):
    model = UNet2(
        input_channels=input_channels,
        n_classes=n_classes,
        bilinear=True
    )
    checkpoint = torch.load(model_path, map_location=('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint)
    return model

@lru_cache(maxsize=2)
def load_trained_model_NETT(model_path, input_channels=1, n_classes=1):
    model = UNetNETT(
        input_channels=input_channels,
        n_classes=n_classes,
        bilinear=True
    )
    checkpoint = torch.load(model_path, map_location=('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint)
    return model

def test_model_performance(model_path1, model_path2, model_path_NETT, npy_file_path, out_max, tau, device='cuda', sample_index=0):
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

        'DL_sk_norm': [],
        'model_item_norm':[],
        'DL_recons':[],
        'DL_steps': [],
        'DL_error':[],
        'beta': [],
        'DL_residual': [],

        'IN-steps': [],
        'IN-recons': [],
        'IN-error': [],
        'IN_residual': [],

        'dc_recons': [],

        'NETT_recons': [],
        'NETT_steps': [],
        'NETT_error': [],
        'NETT_residual': []
    }
    
    print(f"Loading custom OOD test sample from {npy_file_path}...")

    # ========================================================
    # 1. 加载真解图像与物理算子
    # ========================================================
    # 加载物理算子 R (7680, 16384)
    R = get_operator()
    test_dataset = get_dataset(npy_file_path)
    gt_np = test_dataset['xk'][sample_index].flatten()
    ydelta_np = test_dataset['ydelta'][sample_index].flatten()
    print(f"Ground truth shape: {gt_np.shape}")
    print(f"ydelta shape: {ydelta_np.shape}")
    # 无噪声的精确投影 (R@x)^2
    Rx = R @ gt_np
    Fx = Rx ** 2
    Fx_2d = Fx.reshape(60, 128)
    y_noise_2d = ydelta_np.reshape(60, 128)


    # 计算精确的噪声水平 delta
    delta = max([np.linalg.norm(y_noise_2d[i, :] - Fx_2d[i, :]) for i in range(60)])
    stop_threshold = tau * delta
    # ========================================================

    results['original_images'].append(gt_np)
    print(f"Sample OOD - Noise Level (delta): {delta:.4e}, Target Stop: {stop_threshold:.4e}")

    model1=load_trained_model(model_path1).to(device).eval()
    modelres=load_trained_model2(model_path2).to(device).eval()
    model_nett=load_trained_model_NETT(model_path_NETT).to(device).eval()
    warmup(model1,device,two_inputs=True)
    warmup(modelres,device)
    warmup(model_nett,device)

    with torch.no_grad():
        for example in TARGET_METHODS:
            begin_method(example,device)
            
            # 初始解必须是 1D 向量 (16384,)
            xk = np.ones((128*128,)) * 0.01
            Rxk = R @ xk
            Fxk = Rxk ** 2
            Fxk_2d = Fxk.reshape(60, 128)
            residual_norm = max([np.linalg.norm(y_noise_2d[i, :] - Fxk_2d[i, :]) for i in range(60)])
            
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

            if example == 'res':
                modelres = load_trained_model2(model_path2)
                modelres = modelres.to(device)
                modelres.eval()

                reserror = np.linalg.norm(xk - gt_np) / np.linalg.norm(gt_np)
                error_list.append(reserror)
                error_list2.append(reserror)
                residual_list.append(residual_norm)
                residual_list2.append(residual_norm)
                
                while k < out_max:
                    k += 1
                    sk, alpha = compute_sk(xk, R, ydelta_np, alpha_start=8, q=0.8, mu_k=0.8)
                    
                    # ---------------- 加入回溯线搜索 (Line Search) ----------------
                    gamma = 0.875
                    for _ in range(6): # 最多回溯 10 次
                        xk_trial = xk + gamma * sk
                        Rxk1 = R @ xk_trial
                        Fxk1 = Rxk1 ** 2
                        Fxk_2d1 = Fxk1.reshape(60, 128)
                        res_trial = max([np.linalg.norm(y_noise_2d[i, :] - Fxk_2d1[i, :]) for i in range(60)])
                        # res_trial = np.linalg.norm((R @ xk_trial)**2 - ydelta_np)
                        if res_trial < residual_norm:   
                            # xk = xk_trial
                            # residual_norm = res_trial
                            break
                        gamma += -0.125
                    # ------------------------------------------------------------------------
                    
                    xk = xk + gamma * sk
                    Rxk = R @ xk
                    Fxk = Rxk ** 2
                    Fxk_2d = Fxk.reshape(60, 128)
                    residual_norm = max([np.linalg.norm(y_noise_2d[i, :] - Fxk_2d[i, :]) for i in range(60)])
                    
                    print(f"IN_step {k}: Residual Norm = {residual_norm:.4e}, Target = {stop_threshold:.4e}")
                    
                    reserror = np.linalg.norm(xk - gt_np) / np.linalg.norm(gt_np)
                    error_list.append(reserror)
                    error_list2.append(reserror)
                    residual_list.append(residual_norm)
                    residual_list2.append(residual_norm)

                    if residual_norm <= stop_threshold:
                        break
                
                results['IN-recons'].append(xk)
                results['IN-steps'].append(k)
                results['IN-error'].append(error_list2)
                results['IN_residual'].append(residual_list2)
                mark_learning_stage('IN',results,R,ydelta_np,delta,stop_threshold,device)
                
                xk_tensor = torch.tensor(xk, dtype=torch.float32, device=device).view(1, 1, 128, 128)
                model_output = modelres(xk_tensor)
                model_output_np = model_output.detach().cpu().numpy().flatten()
                xk1 = xk + 0.65*model_output_np
                Rxk1 = R @ xk1
                Fxk1 = Rxk1 ** 2
                Fxk1_2d = Fxk1.reshape(60, 128)
                residual_norm = max([np.linalg.norm(y_noise_2d[i, :] - Fxk1_2d[i, :]) for i in range(60)])
                
                reserror = np.linalg.norm(xk1 - gt_np) / np.linalg.norm(gt_np)
                error_list.append(reserror)
                residual_list.append(residual_norm)
                print(f"Residual Norm = {residual_norm:.4e}")

                results['res_recons'].append(xk1)
                results['res_steps'].append(k+1)
                results['res_error'].append(error_list)
                results['res_residual'].append(residual_list)
                mark_learning_stage('IN-RES',results,R,ydelta_np,delta,stop_threshold,device)

                R_pinv = get_pinv()
                xk_dc = xk1 - R_pinv @ (R @ xk1 - R @ xk)
                Rxk2 = R @ xk_dc
                Fxk2 = Rxk2 ** 2
                Fxk2_2d = Fxk2.reshape(60, 128)
                dc_residual_norm = max([np.linalg.norm(y_noise_2d[i, :] - Fxk2_2d[i, :]) for i in range(60)])
                dc_error = np.linalg.norm(xk_dc - gt_np) / np.linalg.norm(gt_np)
                print(f"dc_Residual Norm = {dc_residual_norm:.4e}, dc_error = {dc_error:.4e}")
                results['dc_recons'].append(xk_dc)
                mark_learning_stage('IN-DC',results,R,ydelta_np,delta,stop_threshold,device)

                

            if example == 'DL_INM':
                model1 = load_trained_model(model_path1)
                model1 = model1.to(device)
                model1.eval()
                
                reserror = np.linalg.norm(xk - gt_np) / np.linalg.norm(gt_np)
                error_list.append(reserror)
                residual_list.append(residual_norm)

                while k < out_max:
                    k += 1
                    sk, alpha = compute_sk(xk, R, ydelta_np, alpha_start=8, q=0.8, mu_k=0.8)

                    # 纯物理步长的回溯线搜索
                    # ---------------- 加入回溯线搜索 (Line Search) ----------------
                    gamma = 0.875
                    for _ in range(6): # 最多回溯 10 次
                        xk_trial = xk + gamma * sk
                        Rxk1 = R @ xk_trial
                        Fxk1 = Rxk1 ** 2
                        Fxk_2d1 = Fxk1.reshape(60, 128)
                        res_trial = max([np.linalg.norm(y_noise_2d[i, :] - Fxk_2d1[i, :]) for i in range(60)])
                        # res_trial = np.linalg.norm((R @ xk_trial)**2 - ydelta_np)
                        if res_trial < residual_norm:
                            # xk = xk_trial
                            # residual_norm = res_trial
                            break
                        gamma += -0.125
                    # ------------------------------------------------------------------------
                    effective_sk = gamma * sk
                    
                    # 获取网络修正项
                    xk_tensor = torch.tensor(xk, dtype=torch.float32, device=device).view(1, 1, 128, 128)
                    sk_tensor = torch.tensor(effective_sk, dtype=torch.float32, device=device).view(1, 1, 128, 128)
                    model_output = model1(xk_tensor, sk_tensor)
                    norm_sk = np.linalg.norm(sk)
                    norm_net = torch.norm(model_output)
                        

                    L_theory = 0.4
                    # 如果网络输出太大，就按比例缩小它，严格满足 ||U_theta|| <= L * ||s_k||
                    scale_factor = torch.clamp(L_theory * norm_sk / (norm_net + 1e-8), max=1.0)
                    model_output = model_output * scale_factor
                    # ====================================================

                    model_norm_list.append(torch.norm(model_output).item())
                    sk_norm_list.append(torch.norm(sk_tensor).item())
                    model_output_np = model_output.detach().cpu().numpy().flatten()

                    Cbeta = 0.002
                    beta = min(Cbeta * residual_norm, 1.0)
                    record_dl_step(k,gamma,alpha,beta,scale_factor,norm_sk,norm_net,model_output,residual_norm)
                    beta_list.append(beta)
                    
                    xk = xk + effective_sk + beta * model_output_np
                    Rxk = R @ xk
                    Fxk = Rxk ** 2
                    Fxk_2d = Fxk.reshape(60, 128)
                    residual_norm = max([np.linalg.norm(y_noise_2d[i, :] - Fxk_2d[i, :]) for i in range(60)])
                    print(f"DL-INM_step {k}: gamma={gamma:.4f}, beta={beta:.4f}, UNet_norm={torch.norm(model_output).item():.4e}, Res_Norm={residual_norm:.4e}")

                    error_res = np.linalg.norm(xk - gt_np) / np.linalg.norm(gt_np)
                    error_list.append(error_res)
                    residual_list.append(residual_norm)

                    if residual_norm <= stop_threshold:
                        break
                
                results['DL_recons'].append(xk)
                results['DL_steps'].append(k)
                results['DL_error'].append(error_list)
                results['DL_sk_norm'].append(sk_norm_list)
                results['beta'].append(beta_list)
                results['model_item_norm'].append(model_norm_list)
                results['DL_residual'].append(residual_list)
                finish_learning('DL-INM',results,R,ydelta_np,delta,stop_threshold,device)


            if example == 'NETT':
                # ============================================================
                # NETT / learned regularization for nonlinear inverse problem
                #
                # Objective:
                #   min_x  1/2 ||F(x) - y_delta||_2^2
                #        + alpha / 2 ||x - D_theta(x)||_2^2
                #
                # where:
                #   F(x) = (R x)^2
                #   D_theta is the trained U-Net denoiser / learned prior
                # ============================================================

                # ------------------------------------------------------------
                # 1. Load trained denoiser D_theta and freeze its parameters
                # ------------------------------------------------------------
                model_nett = load_trained_model_NETT(model_path_NETT)
                model_nett = model_nett.to(device)
                model_nett.eval()

                for p in model_nett.parameters():
                    p.requires_grad_(False)

                # ------------------------------------------------------------
                # 2. Prepare numpy arrays with consistent column-vector shape
                # ------------------------------------------------------------
                xk_init_np = xk.reshape(-1, 1)
                gt_col = gt_np.reshape(-1, 1)

                y_delta_col = ydelta_np.reshape(-1, 1)
                y_delta_2d = y_delta_col.reshape(60, 128)

                # Initial error and residual
                reserror = np.linalg.norm(xk_init_np - gt_col) / np.linalg.norm(gt_col)
                error_list.append(reserror)
                residual_list.append(residual_norm)

                # ------------------------------------------------------------
                # 3. Convert R, y_delta, and x to torch tensors
                # ------------------------------------------------------------
                R_tensor = get_torch_operator(device)
                y_delta_tensor = torch.as_tensor(y_delta_col, dtype=torch.float32, device=device)

                # x is the optimization variable in NETT.
                x_var = torch.tensor(
                    xk_init_np,
                    dtype=torch.float32,
                    device=device,
                    requires_grad=True
                )

                # ------------------------------------------------------------
                # 4. NETT hyperparameters
                # ------------------------------------------------------------
                lr_nett = 5e-3
                if k <= 100:
                    alpha_current = 0.05
                elif k <= 200:
                    alpha_current = 0.1
                else:
                    alpha_current = 0.5

                # Whether to impose box constraint after each optimizer step.
                # For your phantom setting, ground truth is nonnegative and mostly in [0, 1].
                use_box_constraint = True
                x_min, x_max = 0.0, 1.0

                optimizer = torch.optim.Adam([x_var], lr=lr_nett)

                # Save best iterate according to discrepancy residual.
                best_x_np = xk_init_np.copy()
                best_error = reserror
                best_residual = residual_norm
                best_step = k

                # ------------------------------------------------------------
                # 5. NETT optimization loop
                # ------------------------------------------------------------
                with torch.enable_grad():
                    # BEFORE_NONLINEAR30: while k < out_max:
                    while k < NETT_MAX:
                        k += 1
                        optimizer.zero_grad()

                        # ---------------- data fidelity term ----------------
                        # F(x) = (R x)^2
                        Rx = torch.matmul(R_tensor, x_var)
                        Fx = Rx ** 2


                        # # ---------------- total NETT objective ----------------
                        # loss = loss_data + alpha_nett * loss_prior

                        # 对观测数据归一化，避免平方 Radon 数据尺度过大
                        y_scale = torch.max(torch.abs(y_delta_tensor)).detach() + 1e-8

                        Rx = torch.matmul(R_tensor, x_var)
                        Fx = Rx ** 2

                        loss_data = 0.5 * torch.mean(((Fx - y_delta_tensor) / y_scale) ** 2)

                        x_img = x_var.view(1, 1, 128, 128)
                        denoised_img = model_nett(x_img)

                        loss_prior = 0.5 * torch.mean((x_img - denoised_img) ** 2)

                        loss = loss_data + alpha_current * loss_prior

                        loss.backward()

                        # Optional but recommended for stability
                        # torch.nn.utils.clip_grad_norm_([x_var], max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_([x_var], max_norm=100.0)

                        optimizer.step()

                        # Projection onto feasible image range.
                        # This is important for the nonlinear squared model F(x)=(Rx)^2.
                        with torch.no_grad():
                            if use_box_constraint:
                                x_var.clamp_(x_min, x_max)
                            else:
                                x_var.clamp_(min=0.0)

                        # --------------------------------------------------------
                        # 6. Evaluation in numpy
                        # --------------------------------------------------------
                        x_current_np = x_var.detach().cpu().numpy().reshape(-1, 1)

                        Rx_np = R @ x_current_np
                        Fx_np = Rx_np ** 2
                        Fx_2d = Fx_np.reshape(60, 128)

                        residual_norm = max([
                            np.linalg.norm(y_delta_2d[i, :] - Fx_2d[i, :])
                            for i in range(60)
                        ])

                        error_res = np.linalg.norm(x_current_np - gt_col) / np.linalg.norm(gt_col)

                        error_list.append(error_res)
                        residual_list.append(residual_norm)

                        # Save best iterate by discrepancy residual
                        if residual_norm < best_residual:
                            best_residual = residual_norm
                            best_error = error_res
                            best_x_np = x_current_np.copy()
                            best_step = k

                        print(
                            f"NETT_step {k}: "
                            f"Loss={loss.item():.4e}, "
                            f"Data={loss_data.item():.4e}, "
                            f"Prior={loss_prior.item():.4e}, "
                            f"Alpha={alpha_current:.2e}, "
                            f"Res_Norm={residual_norm:.4e}, "
                            f"Error={error_res:.4e}, "
                            f"Target={stop_threshold:.4e}"
                        )

                        # Discrepancy principle stopping
                        if residual_norm <= stop_threshold:
                            break

                # ------------------------------------------------------------
                # 7. Save final result
                # ------------------------------------------------------------
                # 如果你希望严格保存最后一次迭代：
                xk_final = x_var.detach().cpu().numpy().reshape(-1, 1)


                final_error = np.linalg.norm(xk_final - gt_col) / np.linalg.norm(gt_col)
                Rx_final = R @ xk_final
                Fx_final_2d = (Rx_final ** 2).reshape(60, 128)
                final_residual = max([
                    np.linalg.norm(y_delta_2d[i, :] - Fx_final_2d[i, :])
                    for i in range(60)
                ])

                print(
                    f"NETT finished: "
                    f"Final_step={k}, "
                    f"Final_Error={final_error:.4e}, "
                    f"Final_Residual={final_residual:.4e}, "
                    f"Best_step={best_step}, "
                    f"Best_Error={best_error:.4e}, "
                    f"Best_Residual={best_residual:.4e}"
                )

                results['NETT_recons'].append(xk_final.reshape(-1))
                results['NETT_steps'].append(k)
                results['NETT_error'].append(error_list)
                results['NETT_residual'].append(residual_list)
                finish_learning('NETT',results,R,ydelta_np,delta,stop_threshold,device)


        print(f"Sample {sample_index}: completed {TARGET_METHODS}")
        print("-" * 50)
            
    return results


def evaluate_and_plot(results, save_dir='test_results'):
    """绘图函数保持不变，自适应调整到只画一个样本"""
    os.makedirs(save_dir, exist_ok=True)
    idx = 0 # 只有一个样本
    
    gt = results['original_images'][0].reshape(128, 128)
    resrecon = results['res_recons'][0].reshape(128, 128)
    recon_DL = results['DL_recons'][0].reshape(128, 128)
    recon_in = results['IN-recons'][0].reshape(128, 128)
    recon_dc = results['dc_recons'][0].reshape(128, 128)
    recon_nett = results['NETT_recons'][0].reshape(128, 128) 
    
    data_range = gt.max() - gt.min()
    if data_range == 0: data_range = 1.0
    
    psnr_res, ssim_res = psnr(gt, resrecon, data_range=data_range), ssim(gt, resrecon, data_range=data_range)
    psnr_DL, ssim_DL = psnr(gt, recon_DL, data_range=data_range), ssim(gt, recon_DL, data_range=data_range)
    psnr_in, ssim_in = psnr(gt, recon_in, data_range=data_range), ssim(gt, recon_in, data_range=data_range)
    psnr_dc, ssim_dc = psnr(gt, recon_dc, data_range=data_range), ssim(gt, recon_dc, data_range=data_range)
    psnr_nett, ssim_nett = psnr(gt, recon_nett, data_range=data_range), ssim(gt, recon_nett, data_range=data_range)

    reserr = results['res_error'][0]
    err_DL = results['DL_error'][0]
    err_IN = results['IN-error'][0]
    err_NETT = results['NETT_error'][0]
    sk_norm = results['DL_sk_norm'][0]
    model_norm = results['model_item_norm'][0]

    residual_IN = results['IN_residual'][0]
    residual_res = results['res_residual'][0]
    residual_DL = results['DL_residual'][0]
    residual_NETT = results['NETT_residual'][0]

    vmin, vmax = gt.min(), gt.max()

    # 图 1：反演结果与绝对误差热力图合集
    fig, axes = plt.subplots(2, 5, figsize=(27, 10))
    
    im1 = axes[0, 0].imshow(recon_in, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f"IN\nPSNR: {psnr_in:.2f} | SSIM: {ssim_in:.4f}", fontsize=16,fontweight='bold')
    fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im2 = axes[0, 3].imshow(resrecon, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 3].set_title(f"IN-RES\nPSNR: {psnr_res:.2f} | SSIM: {ssim_res:.4f}", fontsize=16,fontweight='bold')
    fig.colorbar(im2, ax=axes[0, 3], fraction=0.046, pad=0.04)

    im3 = axes[0, 2].imshow(recon_dc, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title(f"IN-DC\nPSNR: {psnr_dc:.2f} | SSIM: {ssim_dc:.4f}", fontsize=16,fontweight='bold')
    fig.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    im4 = axes[0, 4].imshow(recon_DL, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 4].set_title(f"DL-INM\nPSNR: {psnr_DL:.2f} | SSIM: {ssim_DL:.4f}", fontsize=16,fontweight='bold')
    fig.colorbar(im4, ax=axes[0, 4], fraction=0.046, pad=0.04)

    im5 = axes[0, 1].imshow(recon_nett, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f"NETT\nPSNR: {psnr_nett:.2f} | SSIM: {ssim_nett:.4f}", fontsize=16,fontweight='bold')
    fig.colorbar(im5, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    err_map_res = np.abs(gt - resrecon)
    err_map_DL = np.abs(gt - recon_DL)
    err_map_in = np.abs(gt - recon_in)
    err_map_dc = np.abs(gt - recon_dc)
    err_map_NETT = np.abs(gt - recon_nett)
    evmax = max(err_map_res.max(), err_map_DL.max(), err_map_in.max(), err_map_dc.max(), err_map_NETT.max())

    im_e1 = axes[1, 0].imshow(err_map_in, cmap='hot', vmin=0, vmax=evmax)
    axes[1, 0].set_title("Error Heatmap (IN)", fontsize=16,fontweight='bold')
    fig.colorbar(im_e1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im_e2 = axes[1, 3].imshow(err_map_res, cmap='hot', vmin=0, vmax=evmax)
    axes[1, 3].set_title("Error Heatmap (IN-RES)", fontsize=16,fontweight='bold')
    fig.colorbar(im_e2, ax=axes[1, 3], fraction=0.046, pad=0.04)

    im_e3 = axes[1, 2].imshow(err_map_dc, cmap='hot', vmin=0, vmax=evmax)
    axes[1, 2].set_title("Error Heatmap (IN-DC)", fontsize=16,fontweight='bold')
    fig.colorbar(im_e3, ax=axes[1, 2], fraction=0.046, pad=0.04)

    im_e4 = axes[1, 1].imshow(err_map_NETT, cmap='hot', vmin=0, vmax=evmax)
    axes[1, 1].set_title("Error Heatmap (NETT)", fontsize=16,fontweight='bold')
    fig.colorbar(im_e4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im_e5 = axes[1, 4].imshow(err_map_DL, cmap='hot', vmin=0, vmax=evmax)
    axes[1, 4].set_title("Error Heatmap (DL-INM)", fontsize=16,fontweight='bold')
    fig.colorbar(im_e5, ax=axes[1, 4], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ood_test_reconstruction_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图 2：单独的相对误差曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(err_IN)+1), err_IN, 'o-', linewidth=3, color='green', label='IN')
    plt.plot(range(1, len(reserr)+1), reserr, 'x-', linewidth=3, color='red', label='IN-RES')
    plt.plot(range(1, len(err_DL)+1), err_DL, 's-', linewidth=3, color='orange', label='DL-INM')
    plt.plot(range(1, len(err_NETT)+1), err_NETT, 'v-', linewidth=3, color='purple', label='NETT')
    plt.xscale('log')
    plt.xlabel("Iteration Step", fontsize=14,fontweight='bold')
    plt.ylabel("Relative Error", fontsize=14,fontweight='bold')
    plt.grid(True, linestyle='--')
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(save_dir, 'ood_test_error_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()


    # 残差
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(residual_IN)+1), residual_IN, 'o-', linewidth=2, color='green', label='IN')
    plt.plot(range(1, len(residual_res)+1), residual_res, 'x-', linewidth=2, color='red', label='IN-RES')
    plt.plot(range(1, len(residual_DL)+1), residual_DL, 's-', linewidth=2, color='orange', label='DL-INM')
    plt.plot(range(1, len(residual_NETT)+1), residual_NETT, 'v-', linewidth=2, color='purple', label='NETT')
    # plt.title(r"Residual Norm Decay: $\max_i \|F_i(x) - y_i^\delta\|$", fontsize=16)
    plt.xlabel("Iteration Step", fontsize=14)
    plt.ylabel("Residual Norm", fontsize=14)
    plt.grid(True, linestyle='--')
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(save_dir, 'ood_test_residual_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # === 图 3：一维截面对比 (1D Profile) ===
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
        nett_slice = recon_nett[row_idx, :]
        
        # 生成像素索引（Y轴的数据）
        pixel_indices = np.arange(len(gt_slice))
        
        # 绘制图表：注意这里将 slice数据 放在X轴位置，pixel_indices 放在Y轴位置
        ax.plot(gt_slice, pixel_indices, 'k--', linewidth=2, label='Ground Truth')
        ax.plot(nett_slice, pixel_indices, 'brown', linewidth=2, linestyle='-.', label='NETT')
        ax.plot(dc_slice, pixel_indices, 'c-', linewidth=2, label='IN-DC')
        ax.plot(res_slice, pixel_indices, 'r-', linewidth=2, alpha=0.8, label='IN-RES')
        ax.plot(dl_slice, pixel_indices, 'b-', linewidth=2, label='DL-INM')
        
        
        # 设置标题和网格
        ax.set_title(f"Row {row_idx}", fontsize=16,fontweight='bold')
        ax.grid(True, linestyle=':')
        
        # 坐标轴标签互换
        ax.set_xlabel("Intensity", fontsize=14,fontweight='bold')
        if i == 0:
            ax.set_ylabel("Pixel Index", fontsize=14,fontweight='bold') # 只有最左边的图显示Y轴标签

        ax.invert_yaxis()

    # 添加全局标题
    # fig.suptitle("1D Cross-section Profiles", fontsize=18)
    
    # 为了避免遮挡曲线，将图例统一放在最后一个子图里（或者你也可以用 bbox_to_anchor 移到图外）
    axes[-1].legend(fontsize=12, loc='best')
    
    # 自动调整子图间距
    plt.tight_layout()
    # tight_layout() 有时会遮挡 suptitle，稍微调整下顶部边距
    plt.subplots_adjust(top=0.88)
    
    # 保存并关闭
    plt.savefig(os.path.join(save_dir, 'ood_test_1d_profile_multi.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # === 图 4：网络干预强度的衰减机制 ===
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Iteration Step', fontsize=14)
    ax1.set_ylabel('Network Norm ||U(s_k)||', color=color, fontsize=14)
    ax1.plot(range(1, len(model_norm)+1), model_norm, 'o-', color=color, linewidth=2, label='||U(s_k)||')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--')

    ax2 = ax1.twinx()  
    color = 'tab:red'
    
    beta_list = results.get('beta',[0]*len(model_norm))[0] 
    ax2.set_ylabel('Algorithm Weight $\\beta_k$', color=color, fontsize=14)  
    ax2.plot(range(1, len(beta_list)+1), beta_list, 's--', color=color, linewidth=2, label='$\\beta_k$')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Dynamical Behavior of the Neural Network Term", fontsize=16)
    fig.tight_layout()  
    plt.savefig(os.path.join(save_dir, 'dynamic_behavior.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved evaluation plots in '{save_dir}'.")


if __name__ == "__main__":
    from run_benchmark import main
    main(["--group", "learning"])
    raise SystemExit(0)
    MODEL_PATH = "models/best.pth"       
    MODEL_PATH_RES = "models/res_final_best.pth"  
    MODEL_PATH_NETT = "models/NETT_final_best.pth"     

    NPY_FILE_PATH = "data/test_dataset.npz"
    
    SAVE_DIR = "test_results_DL"            

    MAX_ITERATIONS = 200                  
    TAU = 3                        
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"=== Starting OOD Evaluation on {DEVICE.upper()} ===")
    
    if not os.path.exists(MODEL_PATH): raise FileNotFoundError(f"未找到: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH_RES): raise FileNotFoundError(f"未找到: {MODEL_PATH_RES}")
    if not os.path.exists(NPY_FILE_PATH): raise FileNotFoundError(f"未找到: {NPY_FILE_PATH}")
    if not os.path.exists('R_limited.npy'): raise FileNotFoundError("未找到 R_limited.npy")

    results = test_model_performance(
        model_path1=MODEL_PATH, 
        model_path2=MODEL_PATH_RES,
        model_path_NETT=MODEL_PATH_NETT,
        npy_file_path=NPY_FILE_PATH, 
        out_max=MAX_ITERATIONS, 
        tau=TAU, 
        device=DEVICE
    )
    
    print("\nGenerating evaluation plots...")
    evaluate_and_plot(results, save_dir=SAVE_DIR)
    
    print(f"\nAll Done! 你的高对比度实验结果已保存在 '{SAVE_DIR}' 文件夹中。")
