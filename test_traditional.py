import torch
import numpy as np
import os
from inference_models import UNet, CustomDataset
from batch_runtime import (compute_sk, begin_method, checkpoint_sample,
    get_operator, get_dataset, get_pinv, learning_batch, warmup, reuse_learning_method, set_current_sample)
from functools import lru_cache
from landweber import landweber_radon
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_tv_chambolle
import matplotlib.pyplot as plt
import time

# 保证 GPU 异步计算不会让计时偏小
def sync_cuda(device):
    if torch.cuda.is_available() and str(device).startswith('cuda'):
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

def test_model_performance(model_path, dataset_path, out_max, tau, device='cuda', num_samples=30, selected_indices=None):
    """
    测试模型性能并与传统方法比较 (完美融合 1D 数学与 2D 深度学习)
    """
    # 加载模型
    model = load_trained_model(model_path)
    model = model.to(device)
    model.eval()
    warmup(model,device,two_inputs=True)
    
    test_dataset = CustomDataset(dataset_path)
    selected_indices = list(range(num_samples)) if selected_indices is None else list(selected_indices)
    
    results = {
        'indices': selected_indices,
        'original_images':[],

        'IN_steps': [],
        'IN_recons':[],
        'IN_error': [],
        'IN_residual': [],
        'IN_time': [],                
        'IN_inner_total': [],         

        'DL_sk_norm': [],
        'DL_model_norm': [],
        'DL_recons':[],
        'DL_steps': [],
        'DL_error':[],
        'DL_residual': [],
        'DL_time': [],               
        'DL_inner_total': [],         
        'DL_scale_factor': [],        

        'Landweber_steps': [],
        'Landweber_recons':[],
        'Landweber_error': [],
        'Landweber_residual': [],
        'Landweber_time': [],

        'TV_steps':[],            
        'TV_recons': [],      
        'TV_error':[],           
        'TV_residual': [],
        'TV_time': [],               
        'TV_inner_total': [],          
        'vis_sk':[],        # 用于保存第 K 步的物理步长
        'vis_unet': [],      # 用于保存第 K 步的网络修正项
        'vis_xk':[]         # (可选) 顺便保存第 K 步更新前的图像作为对照

    }
    
    print(f"Testing {num_samples} samples...")

    # 加载物理算子 R (7680, 16384)
    R = get_operator()
    
    with torch.no_grad():
        for idx in selected_indices:

            set_current_sample(idx)
            raw_dataset = get_dataset(dataset_path)
            gt_np = raw_dataset['xk'][idx].reshape(-1).astype(np.float64)
            ydelta_np = raw_dataset['ydelta'][idx].reshape(-1).astype(np.float64)

            gt_2d = gt_np.reshape(128, 128)
            plt.figure(figsize=(6, 6))
            plt.imshow(gt_2d, cmap='viridis', vmin=0, vmax=1.0)
            plt.colorbar()
            plt.axis('off')
            os.makedirs('data', exist_ok=True)
            os.makedirs('results/traditional/plots',exist_ok=True)
            plt.savefig(f'results/traditional/plots/ground_truth_{idx}.png', dpi=150, bbox_inches='tight')
            plt.close()

            Fx = R @ gt_np
            delta = np.linalg.norm(Fx - ydelta_np)
            # 停止阈值 (标量)
            stop_threshold = tau * delta

            results['original_images'].append(gt_np)

            print(f"Sample {idx} - Noise Level (delta): {delta:.4e}, Target Stop: {stop_threshold:.4e}")

            for example in ['IN', 'DL-INM', 'Landweber', 'IN-TV']:

                begin_method(example)
                sync_cuda(device)
                start_time = time.perf_counter()
                # 初始解必须是 1D 向量 (16384,)，以适配物理数学运算
                xk = np.ones((128*128,)) * 0.01
                Fxk = R @ xk
                residual_norm = np.linalg.norm(Fxk - ydelta_np)
                print(f"step {0}: Residual Norm = {residual_norm:.4e}")
                
                k = 0
                error_list =[]
                sk_norm_list = []
                model_norm_list =[]
                residual_list = []
                inner_iter_total = 0
                scale_factor_list = []

                print(f"\n--- Running method: {example} ---")

                if example == 'IN':
                    reuse_learning_method(results, idx, 'IN')

                if example == 'DL-INM':
                    reuse_learning_method(results, idx, 'DL')

                if example == 'Landweber':
                    step_size = 1e-4
                    im_LI, res_history, error_list1, niter = landweber_radon(xk, gt_np, R, ydelta_np, tau, delta, step_size, 2000, early_stop=True)
                    sync_cuda(device)
                    elapsed_time = time.perf_counter() - start_time

                    results['Landweber_recons'].append(im_LI)
                    results['Landweber_steps'].append(niter)
                    results['Landweber_error'].append(error_list1)
                    results['Landweber_residual'].append(res_history)
                    results['Landweber_time'].append(elapsed_time)
                    print(
                    f"[Landweber] "
                    f"iterations={niter}, "
                    f"time={elapsed_time:.4f} s"
                    )


                if example == 'IN-TV':
                    # TV 惩罚权重，过大图像会变成色块，过小起不到保边缘去伪影效果
                    # 对于归一化到[0, 1] 附近的图像，通常在 0.01 ~ 0.1 之间调整
                    tv_weight = 0.03
                    error_TV = np.linalg.norm(xk - gt_np) / np.linalg.norm(gt_np)
                    error_list.append(error_TV)
                    residual_list.append(residual_norm)

                    while k < 30:
                        k += 1
                        sk, alpha, inner_count = compute_sk(xk, R, ydelta_np, alpha_start=8, q=0.8, mu_k=0.8)
                        inner_iter_total += inner_count

                        # ---------------- 加入回溯线搜索 (Line Search) ----------------
                        gamma = 0.875
                        for _ in range(6): # 最多回溯 10 次
                            xk_trial = xk + gamma * sk
                            Fxk1 = R @ xk_trial
                            res_trial = np.linalg.norm(Fxk1 - ydelta_np)
                            if res_trial < residual_norm:
                                break
                            gamma += -0.125
                        # ------------------------------------------------------------------------
                        
                        # 2. 物理更新 (Forward Step)
                        xk_half = xk + gamma * sk
                        
                        # 3. TV 近端算子降噪 (Backward / Proximal Step)
                        xk_half_2d = xk_half.reshape(128, 128)
                        xk_new_2d = denoise_tv_chambolle(xk_half_2d, weight=tv_weight, max_num_iter=50)
                        xk = xk_new_2d.flatten()
                        
                        Fxk = R @ xk
                        residual_norm = np.linalg.norm(Fxk - ydelta_np)
                        
                        print(f"TV_step {k}: gamma={gamma:.4f}, Residual Norm = {residual_norm:.4e}")

                        error_TV = np.linalg.norm(xk - gt_np) / np.linalg.norm(gt_np)
                        error_list.append(error_TV)
                        residual_list.append(residual_norm)

                        if residual_norm <= stop_threshold: 
                            break
                    sync_cuda(device)
                    elapsed_time = time.perf_counter() - start_time
                    
                    results['TV_recons'].append(xk)
                    results['TV_steps'].append(k)
                    results['TV_error'].append(error_list)
                    results['TV_residual'].append(residual_list)

                    results['TV_time'].append(elapsed_time)
                    results['TV_inner_total'].append(inner_iter_total)

                    print(f"[IN-TV] outer={k}, "f"inner_total={inner_iter_total}, "f"time={elapsed_time:.4f} s")

                # =======================================================

            
            # 打印最终对比
            print(f"Sample {idx} Done. Final Error - IN: {results['IN_error'][-1][-1]:.4f}, UNet: {results['DL_error'][-1][-1]:.4f}, Landweber: {results['Landweber_error'][-1][-1]:.4f}, TV: {results['TV_error'][-1][-1]:.4f}")
            print("-" * 50)
            print("\n" + "=" * 70)
            print("COMPUTATIONAL COST SUMMARY")
            print("=" * 70)

            print(
                f"IN        : "
                f"outer={results['IN_steps'][-1]}, "
                f"inner={results['IN_inner_total'][-1]}, "
                f"time={results['IN_time'][-1]:.4f}s"
            )

            print(
                f"DL-INM    : "
                f"outer={results['DL_steps'][-1]}, "
                f"inner={results['DL_inner_total'][-1]}, "
                f"time={results['DL_time'][-1]:.4f}s"
            )

            print(
                f"Landweber : "
                f"iter={results['Landweber_steps'][-1]}, "
                f"time={results['Landweber_time'][-1]:.4f}s"
                )

            print(
                f"IN-TV     : "
                f"outer={results['TV_steps'][-1]}, "
                f"inner={results['TV_inner_total'][-1]}, "
                f"time={results['TV_time'][-1]:.4f}s"
            )

            print(
             "DL scale factors:",
             [f"{x:.4f}" for x in results['DL_scale_factor'][-1]]
            )

            print("=" * 70)
            
            checkpoint_sample("traditional",idx,results,len(results["original_images"])-1,dataset_path,tau)

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
        recon_IN = results['IN_recons'][i].reshape(128, 128)
        recon_TV = results['TV_recons'][i].reshape(128, 128)
        recon_res = results['DL_recons'][i].reshape(128, 128)
        
        data_range = gt.max() - gt.min()
        if data_range == 0: data_range = 1.0
        
        psnr_IN, ssim_IN = psnr(gt, recon_IN, data_range=data_range), ssim(gt, recon_IN, data_range=data_range)
        psnr_landweber, ssim_landweber = psnr(gt, recon_Landweber, data_range=data_range), ssim(gt, recon_Landweber, data_range=data_range)
        psnr_TV, ssim_TV = psnr(gt, recon_TV, data_range=data_range), ssim(gt, recon_TV, data_range=data_range)
        psnr_res, ssim_res = psnr(gt, recon_res, data_range=data_range), ssim(gt, recon_res, data_range=data_range)
        
        err_landweber = results['Landweber_error'][i]
        err_IN = results['IN_error'][i]
        err_TV = results['TV_error'][i]
        err_res = results['DL_error'][i]
        sk_norm = results['DL_sk_norm'][i]
        model_norm = results['DL_model_norm'][i]

        residual_Landweber = results['Landweber_residual'][i]
        residual_IN = results['IN_residual'][i]
        residual_res = results['DL_residual'][i]
        residual_TV = results['TV_residual'][i]

        vmin, vmax = gt.min(), gt.max()

        # =====================================================
        # 图 1：反演结果与绝对误差热力图合集 (2行5列)
        # =====================================================
        fig, axes = plt.subplots(2, 4, figsize=(25, 10))
        # fig.suptitle(f"Reconstruction & Absolute Error", fontsize=18, fontweight='bold')
        
        # --- 第一行：反演图 ---
        # im0 = axes[0, 0].imshow(gt, cmap='viridis', vmin=vmin, vmax=vmax)
        # axes[0, 0].set_title("Ground Truth", fontsize=14,fontweight='bold')
        # fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        im1 = axes[0, 0].imshow(recon_Landweber, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title(f"Landweber\nPSNR: {psnr_landweber:.2f} | SSIM: {ssim_landweber:.4f}", fontsize=14,fontweight='bold')
        fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        im2 = axes[0, 1].imshow(recon_IN, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title(f"IN\nPSNR: {psnr_IN:.2f} | SSIM: {ssim_IN:.4f}", fontsize=14,fontweight='bold')
        fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im3 = axes[0, 2].imshow(recon_TV, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, 2].set_title(f"IN-TV\nPSNR: {psnr_TV:.2f} | SSIM: {ssim_TV:.4f}", fontsize=14,fontweight='bold')
        fig.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        im4 = axes[0, 3].imshow(recon_res, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, 3].set_title(f"DL-INM\nPSNR: {psnr_res:.2f} | SSIM: {ssim_res:.4f}", fontsize=14,fontweight='bold')
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
        plt.plot(range(1, len(err_landweber)+1), err_landweber, 'x-', linewidth=3, color='red', label='Landweber')
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
        plt.plot(range(1, len(residual_IN)+1), residual_IN, 'o-', linewidth=2, label='IN')
        plt.plot(range(1, len(residual_TV)+1), residual_TV, '^-', linewidth=2, color='green', label='IN-TV')
        plt.plot(range(1, len(residual_res)+1), residual_res, 's-', linewidth=2, color='orange', label='DL-INM')
        # plt.title(r"Residual Norm Decay: $\|F(x) - y^\delta\|$", fontsize=16)
        plt.xlabel("Iteration Step", fontsize=14,fontweight='bold')
        plt.ylabel("Residual Norm", fontsize=14,fontweight='bold')
        # 注意：由于 Landweber 迭代可能多达几千步，而 IN 只有几十步，X轴画在一起可能会把 IN 挤到左边。
        # 如果曲线显示不佳，使用对数坐标轴
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
        plt.plot(range(1, len(sk_norm)+1), sk_norm, '^-', linewidth=2, color='blue', label='Physics Step (sk)')
        plt.plot(range(1, len(model_norm)+1), model_norm, 'd-', linewidth=2, color='red', label='UNet Correction')
        plt.title(f"Physics vs Network Contribution", fontsize=16)
        plt.xlabel("Iteration Step", fontsize=14)
        plt.ylabel("L2 Norm", fontsize=14)
        plt.yscale('log')
        plt.grid(True, linestyle='--')
        plt.legend(fontsize=12)
        save_path_norm = os.path.join(save_dir, f'contribution_curve_sample_{idx}.png')
        plt.savefig(save_path_norm, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved evaluation plots for sample {idx} in '{save_dir}'.")

def plot_intermediate_components(results, target_k=1, save_dir='test_results'):
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
    # 配置参数设置
    MODEL_PATH = "models/best.pth"       # 训练好的模型路径 (确保模型已经存在)
    DATASET_PATH = "data/test_dataset.npz"
    SAVE_DIR = "results/traditional/plots"

    MAX_ITERATIONS = 50
    TAU = 1.5
    NUM_SAMPLES = 30
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

    print("\nGenerating evaluation plots...")
    evaluate_and_plot(results, save_dir=SAVE_DIR)
    
    print("\nGenerating component analysis plot...")
    plot_intermediate_components(results, target_k=1, save_dir=SAVE_DIR)
    
    print(f"\nAll Done! Please check the '{SAVE_DIR}' folder for visualizations.")
