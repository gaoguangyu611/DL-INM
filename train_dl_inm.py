"""DL-INM linear CT network training (raw Newton step + separate alpha/gamma).

Requires train_dataset.npz / val_dataset.npz produced by
`generate_linear_dataset_raw_sk_gamma.py` (including `sk` and `gamma`).
The training objective is exactly (1/M)*sum_m ||x_m + alpha_m*s_m + U_theta(x_m,s_m)-gt_m||_2^2,
where U_theta is the per-sample norm-bounded correction. Beta is NOT part of this loss.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import time


# DoubleConv：双卷积块，包含两个连续的卷积层
class DoubleConv(nn.Module):
    """
    A module that performs two consecutive convolution operations, each
    followed by a ReLU activation.

    This module implements a double convolution layer block, where each
    convolution is a 3x3 convolution followed by ReLU activation.

    Parameters:
    -----------
    in_channels : int
        The number of input channels.
    out_channels : int
        The number of output channels.
    mid_channels : int, optional
        The number of channels after the first convolution. If not specified,
        defaults to out_channels.

    Methods:
    --------
    forward(x: torch.Tensor) -> torch.Tensor
        Applies the double convolution block to the input tensor and returns
        the output.
    """

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None
    ):

        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


# Down：下采样模块，包含最大池化和双卷积
class Down(nn.Module):
    """
    A module that performs downsampling followed by a double convolution.

    This module first downsamples the input using max pooling and then applies
    a double convolution layer.

    Parameters:
    -----------
    in_channels : int
        The number of input channels.
    out_channels : int
        The number of output channels.

    Methods:
    --------
    forward(x: torch.Tensor) -> torch.Tensor
        Applies max pooling followed by a double convolution and returns
        the output.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


# Up：上采样模块，支持双线性插值或转置卷积
class Up(nn.Module):
    """
    A module that performs upsampling followed by a double convolution.

    This module first upsamples the input using either bilinear upsampling or
    transposed convolution, and then applies a double convolution layer.

    Parameters:
    -----------
    in_channels : int
        The number of input channels.
    out_channels : int
        The number of output channels.
    bilinear : bool, optional
        Whether to use bilinear upsampling (True) or transposed convolution
        (False). Defaults to True.

    Methods:
    --------
    forward(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor
        Upsamples the input tensor `x1`, concatenates it with `x2`, and
        applies a double convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of
        # channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    A module that performs a 1x1 convolution to map the features to the
    desired number of output channels.

    Parameters:
    -----------
    in_channels : int
        The number of input channels.
    out_channels : int
        The number of output channels.

    Methods:
    --------
    forward(x: torch.Tensor) -> torch.Tensor
        Applies the 1x1 convolution and returns the output.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# UNet 类：主网络类，支持多种约束和正则化方法
class UNet(nn.Module):
    

    def __init__(
        self,
        input1_channels: int,      # 第一个输入的通道数
        input2_channels: int,      # 第二个输入的通道数
        n_classes: int,            # 输出通道数
        radon_full=None,
        bilinear: bool = True,
    ):

        super(UNet, self).__init__()
        # 计算总输入通道数
        self.input1_channels = input1_channels
        self.input2_channels = input2_channels
        self.n_channels = input1_channels + input2_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 网络结构保持不变，只是输入通道数调整
        self.inc = DoubleConv(self.n_channels, 64)  # 注意这里使用总通道数
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)


    def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = torch.cat([X1, X2], dim=1)

        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        res = self.outc(x)

        return res


class CustomDataset(Dataset):
    """Dataset generated by generate_linear_dataset_raw_sk_gamma.py.

    IMPORTANT:
      sk    = raw/unrelaxed Newton direction s_k
      gamma = outer line-search coefficient alpha_k (NOT neural projection gamma)

    The stored measurement ydelta is not needed for the loss in this experiment.
    """

    def __init__(self, npz_file_path, img_size=128):
        with np.load(npz_file_path) as data:
            required = ("xk", "xkdelta", "sk", "gamma")
            missing = [k for k in required if k not in data.files]
            if missing:
                raise ValueError(
                    f"{npz_file_path} is missing {missing}. Regenerate the dataset "
                    "with the RAW sk + separate gamma generator; do NOT use the old "
                    "dataset whose sk already includes the line-search factor."
                )
            self.x_true = data["xk"].astype(np.float32)
            self.x_iter = data["xkdelta"].astype(np.float32)
            self.sk_raw = data["sk"].astype(np.float32)
            self.alpha = np.asarray(data["gamma"], dtype=np.float32).reshape(-1)

        self.length = len(self.x_true)
        self.img_size = img_size
        if self.length == 0:
            raise ValueError(f"{npz_file_path}: dataset is empty")
        if not (len(self.x_iter) == len(self.sk_raw) == len(self.alpha) == self.length):
            raise ValueError(f"{npz_file_path}: inconsistent number of rows")
        for name, arr in (("xk", self.x_true), ("xkdelta", self.x_iter),
                          ("sk", self.sk_raw)):
            if np.prod(arr.shape[1:]) != img_size * img_size:
                raise ValueError(f"{npz_file_path}: {name} has invalid shape {arr.shape}")
        if not np.all(np.isfinite(self.alpha)) or not np.all(
            (self.alpha > 0) & (self.alpha <= 1)
        ):
            raise ValueError(f"{npz_file_path}: gamma/alpha must be finite in (0, 1]")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        shape = (1, self.img_size, self.img_size)
        x_true = torch.from_numpy(self.x_true[idx].reshape(shape))
        x_iter = torch.from_numpy(self.x_iter[idx].reshape(shape))
        sk_raw = torch.from_numpy(self.sk_raw[idx].reshape(shape))
        alpha = torch.tensor(self.alpha[idx], dtype=torch.float32)
        return x_true, x_iter, sk_raw, alpha


# 清空GPU缓存（仅在 CUDA 可用时）
if torch.cuda.is_available():
    torch.cuda.empty_cache()


def get_model(training_params, device, input1_channels=1, input2_channels=1, n_classes=1):
    """
    获取模型、优化器和损失函数
    """
    # 初始化模型
    model = UNet(
        input1_channels=input1_channels,
        input2_channels=input2_channels,
        n_classes=n_classes,
        bilinear=training_params.get('bilinear', True)
    )
    
    # 移动模型到指定设备
    model = model.to(device)
    
    # 定义优化器
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=training_params.get('learning_rate', 1e-3),
        weight_decay=training_params.get('weight_decay', 0)
    )
    
    # L2 loss is computed exactly as the formula: mean across samples of squared L2 norms.
    # Hence no pixel-wise nn.MSELoss is returned here.
    return model, optimizer


def plot_losses(train_losses, test_losses, save_path):
    """
    绘制训练和测试损失曲线
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, test_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss plot saved to {save_path}")


def get_data_loader(config):
    """
    直接从两个独立的 npz 文件加载训练集和验证集，彻底杜绝数据泄露。
    """
    # 分别创建训练和验证 Dataset
    train_dataset = CustomDataset(config.train_dataset_path, img_size=config.image_size)
    val_dataset = CustomDataset(config.val_dataset_path, img_size=config.image_size)
    
    print(f"训练集大小 (步数): {len(train_dataset)}")
    print(f"验证集大小 (步数): {len(val_dataset)}")
    
    # 创建 DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, # 训练集需要打乱
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,   # 验证集不需要打乱
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


# 核心训练逻辑，包含数据加载、前向传播、损失计算、反向传播和参数更新
def bounded_correction(raw_output, sk_raw, L=0.4, epsilon0=1e-8):
    """Per-sample bounded operator U_theta(x,s) = projection_scale * N_theta(x,s).

    Each row represents ONE image; never take an L2 norm over the entire batch.
    Projection is differentiable (no detach / no torch.no_grad in training).
    """
    norm_s = torch.linalg.vector_norm(sk_raw.flatten(1), dim=1).view(-1, 1, 1, 1)
    norm_net = torch.linalg.vector_norm(raw_output.flatten(1), dim=1).view(-1, 1, 1, 1)
    project_scale = torch.clamp(L * norm_s / (norm_net + epsilon0), max=1.0)
    return project_scale * raw_output


def reconstruction_and_loss(model, x_true, x_iter, sk_raw, alpha, L, epsilon0):
    """Implements 1/M sum_m ||x_m + alpha_m*s_m + U_theta(x_m,s_m) - gt_m||_2^2.

    NO beta multiplier: the requested training formula intentionally omits beta.
    Important: sk_raw and alpha are saved SEPARATELY by the new generator.
    """
    raw_output = model(x_iter, sk_raw)
    correction = bounded_correction(raw_output, sk_raw, L=L, epsilon0=epsilon0)
    alpha = alpha.reshape(-1, 1, 1, 1)
    prediction = x_iter + alpha * sk_raw + correction

    # Exact paper formula: mean of *per-image squared Euclidean norms*.
    # This is num_pixels times torch.nn.functional.mse_loss(prediction, x_true).
    squared_l2_per_image = (prediction - x_true).square().flatten(1).sum(dim=1)
    loss = squared_l2_per_image.mean()
    return loss


def train(config):
    """Train the dual-input UNet with the projected correction in the requested loss."""
    if config.training_params.get("loss_fn") != "MSE":
        raise ValueError("This experiment implements exactly the squared L2 loss; use loss_fn='MSE'.")

    best_test_loss = float("inf")
    best_model_params = None
    best_epoch = 0

    model, optimizer = get_model(
        config.training_params, config.device,
        config.input1_channels, config.input2_channels, config.n_classes
    )
    TrainDataGen, TestDataGen = get_data_loader(config)
    torch.cuda.empty_cache()
    train_loss_list = np.zeros(config.epochs)
    test_loss_list = np.zeros(config.epochs)

    print("Starting training with x + alpha * raw_sk + U_theta(x, raw_sk)")
    print(f"L={config.bound_L}, epsilon0={config.projection_epsilon}; beta is not used")
    print(f"Train states: {len(TrainDataGen.dataset)}, Val states: {len(TestDataGen.dataset)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    if config.device.type == "cuda":
        torch.cuda.synchronize(config.device)
    training_start = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_sample_count = 0
        for batch_idx, (x_true, x_iter, sk_raw, alpha) in enumerate(TrainDataGen):
            x_true = x_true.to(config.device, non_blocking=True)
            x_iter = x_iter.to(config.device, non_blocking=True)
            sk_raw = sk_raw.to(config.device, non_blocking=True)
            alpha = alpha.to(config.device, non_blocking=True)

            loss = reconstruction_and_loss(
                model, x_true, x_iter, sk_raw, alpha,
                L=config.bound_L, epsilon0=config.projection_epsilon
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = x_true.size(0)
            train_loss_sum += loss.item() * batch_size
            train_sample_count += batch_size
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, batch {batch_idx}, loss: {loss.item():.6f}")

        model.eval()
        val_loss_sum = 0.0
        val_sample_count = 0
        with torch.no_grad():
            for x_true, x_iter, sk_raw, alpha in TestDataGen:
                x_true = x_true.to(config.device, non_blocking=True)
                x_iter = x_iter.to(config.device, non_blocking=True)
                sk_raw = sk_raw.to(config.device, non_blocking=True)
                alpha = alpha.to(config.device, non_blocking=True)

                loss = reconstruction_and_loss(
                    model, x_true, x_iter, sk_raw, alpha,
                    L=config.bound_L, epsilon0=config.projection_epsilon
                )
                batch_size = x_true.size(0)
                val_loss_sum += loss.item() * batch_size
                val_sample_count += batch_size

        train_loss = train_loss_sum / train_sample_count
        test_loss = val_loss_sum / val_sample_count
        train_loss_list[epoch - 1] = train_loss
        test_loss_list[epoch - 1] = test_loss

        if test_loss < best_test_loss:
            best_epoch = epoch
            best_test_loss = test_loss
            best_model_params = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            os.makedirs("models", exist_ok=True)
            torch.save(best_model_params, "models/best.pth")
            print(f"New best model saved at epoch {epoch}: val loss {test_loss:.8f}")

        print(
            f"Epoch [{epoch}/{config.epochs}], Train Loss: {train_loss:.8f}, "
            f"Validation Loss: {test_loss:.8f}"
        )
        if epoch % config.plot_freq == 0 or epoch == 1:
            os.makedirs("losses", exist_ok=True)
            plot_losses(
                train_loss_list[:epoch], test_loss_list[:epoch],
                f"losses/loss_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )

    if best_model_params is None:
        raise RuntimeError("Training completed without any valid validation loss")
    torch.save(best_model_params, "models/final_best.pth")
    if config.device.type == "cuda":
        torch.cuda.synchronize(config.device)
    training_seconds = time.perf_counter() - training_start
    with open("training_time.txt", "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"训练耗时: {training_seconds:.2f} 秒 "
            f"({training_seconds / 60:.2f} 分钟) "
            f"loss=x+alpha*raw_sk+U, L={config.bound_L}\n"
        )
    print(f"Training completed. Best epoch {best_epoch}; val loss {best_test_loss:.8f}")
    print(f"Training time: {training_seconds:.2f} seconds")
    print("Saved to models/final_best.pth")


# 配置类
class Config:
    def __init__(self):
        # 数据集配置
        self.train_dataset_path = "data/train_dataset.npz"  # 训练集文件路径
        self.val_dataset_path = "data/val_dataset.npz" 

        self.input1_channels = 1  # xkdelta的通道数
        self.input2_channels = 1  # sk的通道数
        self.n_classes = 1        # 输出通道数 (xkdelta)
        self.batch_size = 8
        # self.train_ratio = 0.8
        self.num_workers = 4
        self.image_size = 128
        
        # 训练配置
        self.epochs = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 其他配置
        self.plot_freq = 50  # 每50个epoch绘制一次损失曲线
        self.bound_L = 0.4
        self.projection_epsilon = 1e-8
        
        # 训练参数
        self.training_params = {
            'learning_rate': 2e-4,
            'weight_decay': 1e-5,
            'bilinear': True,
            'loss_fn': 'MSE'   # 固定为公式中的平方 L2 损失
        }


if __name__ == "__main__":
    # 创建配置
    config = Config()

    # 开启 cuDNN 基准测试，网络结构不变时大幅提升卷积性能
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # 检查数据文件是否存在（现在需要检查两个文件）
    if not os.path.exists(config.train_dataset_path):
        raise FileNotFoundError(f"Training dataset file {config.train_dataset_path} not found!")
    if not os.path.exists(config.val_dataset_path):
        raise FileNotFoundError(f"Validation dataset file {config.val_dataset_path} not found!")
    
    print("Training configuration:")
    print(f"- Device: {config.device}")
    print(f"- Epochs: {config.epochs}")
    print(f"- Batch size: {config.batch_size}")
    print(f"- Learning rate: {config.training_params['learning_rate']}")
    print("- Model: U-Net inputs (xkdelta, raw sk) -> bounded U_theta")
    print("- Loss: ||xkdelta + gamma * raw_sk + U_theta - x_true||_2^2, averaged per sample")
    print("=" * 60)
    
    # 创建保存目录
    os.makedirs("models", exist_ok=True)
    os.makedirs("losses", exist_ok=True)

    train(config=config)
    
    print("All models trained successfully!")
