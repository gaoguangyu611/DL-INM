import torch
import numpy as np
import os
from train_dl_inm import UNet, CustomDataset
from runtime import (compute_sk, get_operator, get_dataset, get_pinv,
    get_torch_operator, warmup, begin_method, record_dl_step, finish_traditional, finish_learning,
    mark_learning_stage)
from functools import lru_cache
from data.landweber import landweber_schl
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_tv_chambolle
import matplotlib.pyplot as plt

TARGET_METHODS = ["IN", "res", "Landweber", "IN-TV"]

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

def test_model_performance(model_path, dataset_path, out_max, tau, device='cuda', num_samples=30, sample_index=0):
    """
    测试模型性能并与传统方法比较 (完美融合 1D 数学与 2D 深度学习)
    """
    # 加载模型
    model = load_trained_model(model_path)
    model = model.to(device)
    model.eval()
    warmup(model,device,two_inputs=True)

    selected_indices = [sample_index]
    
    results = {
        'indices': selected_indices,
        'original_images':[],
        'Inexact_Newton_steps': [],
        'Inexact_Newton_recons':[],
        'error_IN': [],
        'residual_IN': [],
        'res_sk_norm': [],
        'model_item_norm': [],
        'res_recons':[],
        'res_steps': [],
        'error_res':[],
        'residual_res': [],
        'Landweber_steps': [],
        'Landweber_recons':[],
        'error_Landweber': [],
        'residual_Landweber': [],
        'TV_steps':[],             # 新增：TV算法迭代步数
        'TV_recons': [],            # 新增：TV算法重建结果
        'error_TV':[],             # 新增：TV算法误差曲线
        'residual_TV': [],
        'vis_sk':[],        # 用于保存第 K 步的物理步长
        'vis_unet': [],      # 用于保存第 K 步的网络修正项
        'vis_xk':[]         # (可选) 顺便保存第 K 步更新前的图像作为对照

    }
    
    print(f"Testing {num_samples} samples...")

    # 加载物理算子 R (7680, 16384)
    R = get_operator()
    test_dataset = CustomDataset(dataset_path)
    
    with torch.no_grad():
        for idx in selected_indices:

            ground_truth, ydelta_tensor, _, _ = test_dataset[idx]
            
            gt_np = ground_truth.detach().cpu().numpy().flatten()    # 变成 (16384,)
            ydelta_np = ydelta_tensor.detach().cpu().numpy().flatten()  # 变成 (7680,)

            gt_2d = gt_np.reshape(128, 128)
            plt.figure(figsize=(6, 6))
            plt.imshow(gt_2d, cmap='viridis', vmin=0, vmax=1.0)
            plt.colorbar()
            plt.axis('off')
            os.makedirs('data', exist_ok=True)

            plt.close()  # Independent batch driver saves a shared dataset montage

            Rx = R @ gt_np
            Fx = Rx ** 2
            Fx_2d = Fx.reshape(60, 128)
            ydelta_2d = ydelta_np.reshape(60, 128)
            delta = max([np.linalg.norm(ydelta_2d[i, :] - Fx_2d[i, :]) for i in range(60)])
            # 停止阈值 (标量)
            stop_threshold = tau * delta

            results['original_images'].append(gt_np)

            print(f"Sample {idx} - Noise Level (delta): {delta:.4e}, Target Stop: {stop_threshold:.4e}")

            for example in TARGET_METHODS:
                begin_method(example,device)
                
                # 初始解必须是 1D 向量 (16384,)，以适配物理数学运算
                xk = np.ones((128*128,)) * 0.01
                Rxk = R @ xk
                Fxk = Rxk ** 2
                Fxk_2d = Fxk.reshape(60, 128)
                residual_norm = max([np.linalg.norm(ydelta_2d[i, :] - Fxk_2d[i, :]) for i in range(60)])
                print(f"step {0}: Residual Norm = {residual_norm:.4e}")
                
                k = 0
                error_list =[]
                sk_norm_list = []
                model_norm_list =[]
                residual_list = []
                
                

                print(f"\n--- Running method: {example} ---")

                if example == 'IN':
                    error_IN = np.linalg.norm(xk - gt_np) / np.linalg.norm(gt_np)
                    error_list.append(error_IN)
                    residual_list.append(residual_norm)
                    while k < out_max:
                        k += 1
                        sk, alpha = compute_sk(xk, R, ydelta_np, alpha_start=8, q=0.8, mu_k=0.8)
                        
                        # # 传统的纯数学更新：全是 1D 向量相加
                        # xk = xk + sk

                        # ---------------- 加入回溯线搜索 (Line Search) ----------------
                        gamma = 0.875
                        for _ in range(6): # 最多回溯 10 次
                            xk_trial = xk + gamma * sk
                            Rxk1 = R @ xk_trial
                            Fxk1 = Rxk1 ** 2
                            Fxk_2d1 = Fxk1.reshape(60, 128)
                            res_trial = max([np.linalg.norm(ydelta_2d[i, :] - Fxk_2d1[i, :]) for i in range(60)])
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
                        residual_norm = max([np.linalg.norm(ydelta_2d[i, :] - Fxk_2d[i, :]) for i in range(60)])
                        
                        # residual_norm = np.linalg.norm(Fxk - ydelta_np)

                        # 计算相对误差 (Relative Error)
                        error_IN = np.linalg.norm(xk - gt_np) / np.linalg.norm(gt_np)
                        error_list.append(error_IN)
                        residual_list.append(residual_norm)

                        print(f"IN_step {k}: gamma = {gamma:.4e}, Residual Norm = {residual_norm:.4e}, Target = {stop_threshold:.4e}, Relative Error = {error_IN:.4e}")

                        if residual_norm <= stop_threshold:
                            break
                    
                    results['Inexact_Newton_recons'].append(xk)
                    results['Inexact_Newton_steps'].append(k)
                    results['error_IN'].append(error_list)
                    results['residual_IN'].append(residual_list)

                if example == 'res':
                    error_res = np.linalg.norm(xk - gt_np) / np.linalg.norm(gt_np)
                    error_list.append(error_res)
                    residual_list.append(residual_norm)
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
                            res_trial = max([np.linalg.norm(ydelta_2d[i, :] - Fxk_2d1[i, :]) for i in range(60)])
                            # res_trial = np.linalg.norm((R @ xk_trial)**2 - ydelta_np)
                            if res_trial < residual_norm:
                                # xk = xk_trial
                                # residual_norm = res_trial
                                break
                            gamma += -0.125
                        # ------------------------------------------------------------------------
                    
                        effective_sk = gamma * sk
                        
                        # 1D NumPy -> 2D PyTorch Tensor (Batch, C, H, W)
                        xk_tensor = torch.tensor(xk, dtype=torch.float32, device=device).view(1, 1, 128, 128)
                        sk_tensor = torch.tensor(effective_sk, dtype=torch.float32, device=device).view(1, 1, 128, 128)

                        # U-Net 介入，输出也是 (1, 1, 128, 128)
                        model_output = model(xk_tensor, sk_tensor)

                        norm_sk = np.linalg.norm(sk)
                        norm_net = torch.norm(model_output)

                        L_theory = 0.4
                        # 如果网络输出太大，就按比例缩小它，严格满足 ||U_theta|| <= L * ||s_k||
                        scale_factor = torch.clamp(L_theory * norm_sk / (norm_net + 1e-8), max=1.0)
                        model_output = model_output * scale_factor
                        # ====================================================

                        # 记录网络输出与物理步长的大小关系
                        model_norm_list.append(torch.norm(model_output).item())
                        sk_norm_list.append(torch.norm(sk_tensor).item())

                        model_output_np = model_output.detach().cpu().numpy().flatten()
                        Cbeta = 0.002
                        beta = min(Cbeta * residual_norm, 1.0)
                        record_dl_step(k,gamma,alpha,beta,scale_factor,norm_sk,norm_net,model_output,residual_norm)

                        TARGET_K = 2  # 可以改成任意想观察的步数，比如 1 或 2
                        
                        if k == TARGET_K:
                            # 提取 2D 形状的物理更新量 (乘以线搜索步长后的真实物理贡献)
                            sk_2d = effective_sk.reshape(128, 128)
                            
                            # 提取 2D 形状的网络更新量 (乘以权重 beta 后的真实网络贡献)
                            unet_2d = (beta * model_output_np).reshape(128, 128)
                            
                            # 顺便把这一步更新前的 xk 也存下来，方便对比
                            xk_2d = xk.copy().reshape(128, 128)
                            
                            results['vis_sk'].append(sk_2d)
                            results['vis_unet'].append(unet_2d)
                            results['vis_xk'].append(xk_2d)
                        # ----------------------------------------------------
                            

                        # 最终更新
                        xk = xk + effective_sk + beta * model_output_np
                        Rxk = R @ xk
                        Fxk = Rxk ** 2
                        Fxk_2d = Fxk.reshape(60, 128)
                        residual_norm = max([np.linalg.norm(ydelta_2d[i, :] - Fxk_2d[i, :]) for i in range(60)])

                        print(f"res_step {k}: gamma={gamma:.4f}, beta={beta:.4f}, UNet_norm={torch.norm(model_output).item():.4e},sk_norm = {torch.norm(sk_tensor).item():.4e}, Res_Norm={residual_norm:.4e}")

                        # 计算相对误差
                        error_res = np.linalg.norm(xk - gt_np) / np.linalg.norm(gt_np)
                        error_list.append(error_res)
                        residual_list.append(residual_norm)

                        if residual_norm <= stop_threshold:
                            break
                    
                    results['res_recons'].append(xk)
                    results['res_steps'].append(k)
                    results['error_res'].append(error_list)
                    results['res_sk_norm'].append(sk_norm_list)
                    results['model_item_norm'].append(model_norm_list)
                    results['residual_res'].append(residual_list)

                if example == 'Landweber':
                    step_size = 2e-7
                    im_LI, res_history, error_list1, niter = landweber_schl(xk, gt_np, R, ydelta_np, tau, delta, step_size, 2000, early_stop=True)
                    
                    results['Landweber_recons'].append(im_LI)
                    results['Landweber_steps'].append(niter)
                    results['error_Landweber'].append(error_list1)
                    results['residual_Landweber'].append(res_history)


                if example == 'IN-TV':
                    # TV 惩罚权重，过大图像会变成色块，过小起不到保边缘去伪影效果
                    # 对于归一化到[0, 1] 附近的图像，通常在 0.01 ~ 0.1 之间调整
                    tv_weight = 0.05
                    error_TV = np.linalg.norm(xk - gt_np) / np.linalg.norm(gt_np)
                    error_list.append(error_TV)
                    residual_list.append(residual_norm)

                    while k < 30:
                        k += 1
                        sk, alpha = compute_sk(xk, R, ydelta_np, alpha_start=8, q=0.8, mu_k=0.8)
                        
                        # ---------------- 加入回溯线搜索 (Line Search) ----------------
                        gamma = 0.875
                        for _ in range(6): # 最多回溯 10 次
                            xk_trial = xk + gamma * sk
                            Rxk1 = R @ xk_trial
                            Fxk1 = Rxk1 ** 2
                            Fxk_2d1 = Fxk1.reshape(60, 128)
                            res_trial = max([np.linalg.norm(ydelta_2d[i, :] - Fxk_2d1[i, :]) for i in range(60)])
                            if res_trial < residual_norm:
                                # xk = xk_trial
                                # residual_norm = res_trial
                                break
                            gamma += -0.125
                        # ------------------------------------------------------------------------
                        
                        # 2. 物理更新 (Forward Step)
                        xk_half = xk + gamma * sk
                        
                        # 3. TV 近端算子降噪 (Backward / Proximal Step)
                        xk_half_2d = xk_half.reshape(128, 128)
                        xk_new_2d = denoise_tv_chambolle(xk_half_2d, weight=tv_weight, max_num_iter=50)
                        xk = xk_new_2d.flatten()
                        
                        Rxk = R @ xk
                        Fxk = Rxk ** 2
                        Fxk_2d = Fxk.reshape(60, 128)
                        residual_norm = max([np.linalg.norm(ydelta_2d[i, :] - Fxk_2d[i, :]) for i in range(60)])
                        
                        
                        print(f"TV_step {k}: gamma={gamma:.4f}, Residual Norm = {residual_norm:.4e}")

                        error_TV = np.linalg.norm(xk - gt_np) / np.linalg.norm(gt_np)
                        error_list.append(error_TV)
                        residual_list.append(residual_norm)

                        if residual_norm <= stop_threshold: 
                            break
                    
                    results['TV_recons'].append(xk)
                    results['TV_steps'].append(k)
                    results['error_TV'].append(error_list)
                    results['residual_TV'].append(residual_list)
                # =======================================================
                finish_traditional(example,results,R,ydelta_np,delta,stop_threshold,device)

            
            # 打印最终对比
            print(f"Sample {idx}: completed {TARGET_METHODS}")
            print("-" * 50)
            
    return results


def evaluate_and_plot(results, save_dir='test_results'):
    """
    1. 反演图和热力图集中在一个 2x5 的图板里。
    2. 误差曲线单独保存。
    3. 步长贡献曲线单独保存。
    """
    os.makedirs(save_dir, exist_ok=True)
    num_samples = len(results['indices'])
    
    for i in range(num_samples):
        idx = results['indices'][i]
        
        gt = results['original_images'][i].reshape(128, 128)
        recon_Landweber = results['Landweber_recons'][i].reshape(128, 128)
        recon_IN = results['Inexact_Newton_recons'][i].reshape(128, 128)
        recon_TV = results['TV_recons'][i].reshape(128, 128)
        recon_res = results['res_recons'][i].reshape(128, 128)
        
        data_range = gt.max() - gt.min()
        if data_range == 0: data_range = 1.0
        
        psnr_IN, ssim_IN = psnr(gt, recon_IN, data_range=data_range), ssim(gt, recon_IN, data_range=data_range)
        psnr_landweber, ssim_landweber = psnr(gt, recon_Landweber, data_range=data_range), ssim(gt, recon_Landweber, data_range=data_range)
        psnr_TV, ssim_TV = psnr(gt, recon_TV, data_range=data_range), ssim(gt, recon_TV, data_range=data_range)
        psnr_res, ssim_res = psnr(gt, recon_res, data_range=data_range), ssim(gt, recon_res, data_range=data_range)
        
        err_landweber = results['error_Landweber'][i]
        err_IN = results['error_IN'][i]
        err_TV = results['error_TV'][i]
        err_res = results['error_res'][i]
        sk_norm = results['res_sk_norm'][i]
        model_norm = results['model_item_norm'][i]

        residual_Landweber = results['residual_Landweber'][i]
        residual_IN = results['residual_IN'][i]
        residual_res = results['residual_res'][i]
        residual_TV = results['residual_TV'][i]

        vmin, vmax = gt.min(), gt.max()

        # =====================================================
        # 图 1：反演结果与绝对误差热力图合集 (2行5列)
        # =====================================================
        fig, axes = plt.subplots(2, 4, figsize=(25, 10))


        im1 = axes[0, 0].imshow(recon_Landweber, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title(f"Landweber\nPSNR: {psnr_landweber:.2f} | SSIM: {ssim_landweber:.4f}", fontsize=14,fontweight='bold')
        fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        im2 = axes[0, 1].imshow(recon_IN, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title(f"IN\nPSNR: {psnr_IN:.2f} | SSIM: {ssim_IN:.4f}", fontsize=14, fontweight='bold')
        fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im3 = axes[0, 2].imshow(recon_TV, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, 2].set_title(f"IN-TV\nPSNR: {psnr_TV:.2f} | SSIM: {ssim_TV:.4f}", fontsize=14, fontweight='bold')
        fig.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        im4 = axes[0, 3].imshow(recon_res, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, 3].set_title(f"DL-INM\nPSNR: {psnr_res:.2f} | SSIM: {ssim_res:.4f}", fontsize=14, fontweight='bold')
        fig.colorbar(im4, ax=axes[0, 3], fraction=0.046, pad=0.04)
        
        # --- 第二行：误差热力图 ---
        err_map_landweber = np.abs(gt - recon_Landweber)
        err_map_IN = np.abs(gt - recon_IN)
        err_map_TV = np.abs(gt - recon_TV)
        err_map_res = np.abs(gt - recon_res)
        evmax = max(err_map_IN.max(), err_map_TV.max(), err_map_res.max(), err_map_landweber.max())

        # axes[1, 0].axis('off') 
        # axes[1, 0].text(0.5, 0.5, "Absolute Error\nHeatmaps", fontsize=16, ha='center', va='center', fontweight='bold')

        im_e1 = axes[1, 0].imshow(err_map_landweber, cmap='hot', vmin=0, vmax=evmax)
        axes[1, 0].set_title("Absolute Error (Landweber)", fontsize=14,fontweight='bold')
        fig.colorbar(im_e1, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im_e2 = axes[1, 1].imshow(err_map_IN, cmap='hot', vmin=0, vmax=evmax)
        axes[1, 1].set_title("Absolute Error (IN)", fontsize=14,fontweight='bold')
        fig.colorbar(im_e2, ax=axes[1, 1], fraction=0.046, pad=0.04)

        im_e3 = axes[1, 2].imshow(err_map_TV, cmap='hot', vmin=0, vmax=evmax)
        axes[1, 2].set_title("Absolute Error (IN-TV)", fontsize=14,fontweight='bold')
        fig.colorbar(im_e3, ax=axes[1, 2], fraction=0.046, pad=0.04)

        im_e4 = axes[1, 3].imshow(err_map_res, cmap='hot', vmin=0, vmax=evmax)
        axes[1, 3].set_title("Absolute Error (DL-INM)", fontsize=14,fontweight='bold')
        fig.colorbar(im_e4, ax=axes[1, 3], fraction=0.046, pad=0.04)

        plt.tight_layout()
        save_path_imgs = os.path.join(save_dir, f'images_heatmaps_sample_{idx}.png')
        plt.savefig(save_path_imgs, dpi=300, bbox_inches='tight')
        plt.close()

        # =====================================================
        # 图 2：单独的相对误差曲线图
        # =====================================================
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(err_landweber)+1), err_landweber, 'x-', linewidth=2, color='red', label='Landweber')
        plt.plot(range(1, len(err_IN)+1), err_IN, 'o-', linewidth=3, label='IN')
        plt.plot(range(1, len(err_TV)+1), err_TV, '^-', linewidth=3, color='green', label='IN-TV')
        plt.plot(range(1, len(err_res)+1), err_res, 's-', linewidth=3, color='orange', label='DL-INM')
        # plt.title(f"Relative Error Decay", fontsize=16)
        plt.xlabel("Iteration Step", fontsize=14,fontweight='bold')
        plt.ylabel("Relative Error", fontsize=14,fontweight='bold')
        # 注意：由于 Landweber 迭代可能多达几千步，而 IN 只有几十步，X轴画在一起可能会把 IN 挤到左边。
        plt.xscale('log') 
        plt.grid(True, linestyle='--')
        plt.legend(fontsize=12)
        save_path_err = os.path.join(save_dir, f'error_curve_sample_{idx}.png')
        plt.savefig(save_path_err, dpi=300, bbox_inches='tight')
        plt.close()

        # =====================================================
        # 单独的相对误差曲线图
        # =====================================================
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(residual_Landweber)+1), residual_Landweber, 'x-', linewidth=2, color='red', label='Landweber')
        plt.plot(range(1, len(residual_IN)+1), residual_IN, 'o-', linewidth=3, label='Inexact Newton (IN)')
        plt.plot(range(1, len(residual_TV)+1), residual_TV, '^-', linewidth=3, color='green', label='IN-TV')
        plt.plot(range(1, len(residual_res)+1), residual_res, 's-', linewidth=3, color='orange', label='DL-INM')
        # plt.title(r"Residual Norm Decay: $\max_i \|F_i(x) - y_i^\delta\|$", fontsize=16)
        plt.xlabel("Iteration Step", fontsize=14,fontweight='bold')
        plt.ylabel("Residual Norm", fontsize=14,fontweight='bold')
        # 注意：由于 Landweber 迭代可能多达几千步，而 IN 只有几十步，X轴画在一起可能会把 IN 挤到左边。
        # 如果曲线显示不佳，你可以取消下面这行注释，使用对数坐标轴
        plt.xscale('log') 
        plt.grid(True, linestyle='--')
        plt.legend(fontsize=12)
        save_path_err = os.path.join(save_dir, f'residual_curve_sample_{idx}.png')
        plt.savefig(save_path_err, dpi=300, bbox_inches='tight')
        plt.close()
        
        # =====================================================
        # 图 3：单独的物理与网络贡献曲线 (仅针对 Res-UNet)
        # =====================================================
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(sk_norm)+1), sk_norm, '^-', linewidth=3, color='blue', label='Physics Step (sk)')
        plt.plot(range(1, len(model_norm)+1), model_norm, 'd-', linewidth=3, color='red', label='UNet Correction')
        # plt.title(f"Physics vs Network Contribution", fontsize=16)
        plt.xlabel("Iteration Step", fontsize=14, fontweight='bold')
        plt.ylabel("L2 Norm", fontsize=14, fontweight='bold')
        plt.yscale('log')
        plt.grid(True, linestyle='--')
        plt.legend(fontsize=12)
        save_path_norm = os.path.join(save_dir, f'contribution_curve_sample_{idx}.png')
        plt.savefig(save_path_norm, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved evaluation plots for sample {idx} in '{save_dir}'.")

def plot_intermediate_components(results, target_k=2, save_dir='test_results'):
    """
    可视化分析第 K 步中，物理步长与网络输出的具体行为差异
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 取出第一个测试样本的保存数据
    xk = results['vis_xk'][0]
    sk = results['vis_sk'][0]
    unet = results['vis_unet'][0]
    
    # 为了对比，计算这一步之后的图像 x_{k+1}
    xk_next = xk + sk + unet
    
    # 物理项和网络项有正有负，我们找到它们绝对值的最大值，用来统一 colorbar
    vmax_update = max(np.abs(sk).max(), np.abs(unet).max())
    vmin_update = -vmax_update

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    # fig.suptitle(f"Component Analysis at Iteration Step k={target_k}", fontsize=20, fontweight='bold')
    
    # 1. 当前图像 x_k
    im0 = axes[0].imshow(xk, cmap='viridis')
    axes[0].set_title(f"Current Image ($x_{target_k-1}$)", fontsize=16,fontweight='bold')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. 物理牛顿步的贡献 (\gamma * s_k)
    im1 = axes[1].imshow(sk, cmap='coolwarm', vmin=vmin_update, vmax=vmax_update)
    axes[1].set_title(r"Physics Step ($\alpha s_k$)", fontsize=16,fontweight='bold')  # 使用原始字符串（raw string）
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. 网络输出的贡献 (\beta * U_\theta(s_k))
    im2 = axes[2].imshow(unet, cmap='coolwarm', vmin=vmin_update, vmax=vmax_update)
    axes[2].set_title(r"Network Correction ($\beta U_\theta(s_k)$)", fontsize=16,fontweight='bold')  # 使用原始字符串
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # 4. 更新后的图像 x_{k+1}
    im3 = axes[3].imshow(xk_next, cmap='viridis')
    axes[3].set_title(f"Updated Image ($x_{target_k}$)", fontsize=16,fontweight='bold')
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'component_analysis_step_{target_k}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved component analysis plot to {save_path}")

# ==========================================
# 主函数入口
# ==========================================
if __name__ == "__main__":
    from run_benchmark import main
    main(["--group", "traditional"])
    raise SystemExit(0)
    # 配置参数设置
    MODEL_PATH = "models/best.pth"       # 训练好的模型路径 (确保模型已经存在)
    DATASET_PATH = "data/test_dataset.npz"    # 测试用的数据集路径
    SAVE_DIR = "test_results"            # 图片结果保存的文件夹
    
    MAX_ITERATIONS = 50                  # 展开和IN的最大迭代步数
    TAU = 3                          # 偏差原理系数，理论要求 tau > 1
    NUM_SAMPLES = 1                      # 随机抽取进行测试和画图的样本数量
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"=== Starting Evaluation on {DEVICE.upper()} ===")
    
    # 1. 检查文件是否存在
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"未找到模型权重文件: {MODEL_PATH}")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"未找到测试数据集: {DATASET_PATH}")
    if not os.path.exists('R_limited.npy'):
        raise FileNotFoundError("未找到 R_limited.npy 物理算子矩阵文件")

    # 2. 执行测试流程，计算误差与各项指标
    results = test_model_performance(
        model_path=MODEL_PATH, 
        dataset_path=DATASET_PATH, 
        out_max=MAX_ITERATIONS, 
        tau=TAU, 
        device=DEVICE,
        num_samples=NUM_SAMPLES
    )
    
    # 3. 评估指标计算，并绘制包含7张图的综合结果面板
    print("\nGenerating evaluation plots...")
    evaluate_and_plot(results, save_dir=SAVE_DIR)
    
    print("\nGenerating component analysis plot...")
    plot_intermediate_components(results, target_k=2, save_dir=SAVE_DIR)
    
    print(f"\nAll Done! Please check the '{SAVE_DIR}' folder for visualizations.")
