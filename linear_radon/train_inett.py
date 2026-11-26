import matplotlib.pyplot as plt
from time import time
import os
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np
import random

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ConstrainedConv2d(nn.Conv2d):
    """
    Conv2d with metadata for nonnegative projection.

    Important:
    This layer does not clamp weights in forward(). Instead, all constrained weights
    are projected by project_nonnegative_() after initialization and after each
    optimizer step. This avoids the zero-gradient issue caused by F.relu(weight).
    """

    def __init__(
        self,
        *args,
        nonnegative_weight: bool = True,
        nonnegative_bias: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.nonnegative_weight = nonnegative_weight
        self.nonnegative_bias = nonnegative_bias

class ConstrainedConvTranspose2d(nn.ConvTranspose2d):
    """ConvTranspose2d with metadata for nonnegative projection."""

    def __init__(
        self,
        *args,
        nonnegative_weight: bool = True,
        nonnegative_bias: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.nonnegative_weight = nonnegative_weight
        self.nonnegative_bias = nonnegative_bias

@torch.no_grad()
def project_nonnegative_(model: nn.Module) -> None:
    """
    Project constrained parameters onto the nonnegative orthant.

    In the iNETT convex U-Net construction, the first convolution connected
    directly to the input is allowed to be free, while subsequent convolutional
    kernels should be nonnegative. The final output bias is also constrained to
    be nonnegative so that Phi_c(x) remains nonnegative.
    """
    for module in model.modules():
        if hasattr(module, "nonnegative_weight") and module.nonnegative_weight:
            module.weight.clamp_(min=0.0)
        if (
            hasattr(module, "nonnegative_bias")
            and module.nonnegative_bias
            and module.bias is not None
        ):
            module.bias.clamp_(min=0.0)


class DoubleConv(nn.Module):
    """
    Two convolutional layers with ReLU activations.

    If is_first_block=True, only the first convolution is unconstrained because
    it is the only convolution directly connected to the input image x. All other
    convolution kernels are constrained to be nonnegative.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        is_first_block: bool = False,
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.conv1 = ConstrainedConv2d(
            in_channels,
            mid_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            nonnegative_weight=not is_first_block,
        )
        self.relu1 = nn.ReLU(inplace=True)
        # 第一层卷积不做非负约束, 否则网络表达能力会太弱。

        self.conv2 = ConstrainedConv2d(
            mid_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            nonnegative_weight=True,
        )
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return x
    
class Down(nn.Module):
    """Downscaling with maxpool followed by convex DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, is_first_block=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.maxpool(x))
    
class Up(nn.Module):
    """Upscaling followed by convex DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels,
                out_channels,
                mid_channels=in_channels // 2,
                is_first_block=False,
            )
        else:
            self.up = ConstrainedConvTranspose2d(
                in_channels,
                in_channels // 2,
                kernel_size=2,
                stride=2,
                bias=False,
                nonnegative_weight=True,
            )
            self.conv = DoubleConv(in_channels, out_channels, is_first_block=False)
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final 1x1 convolution.

    Both the final weight and final bias are projected to be nonnegative. This
    helps keep Phi_c(x) nonnegative when previous feature maps are nonnegative.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConstrainedConv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=True,
            nonnegative_weight=True,
            nonnegative_bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UNetinett(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, bilinear=True, a=1e-4, init_std_scale=0.05):
        super().__init__()
        self.a = a
        self.init_std_scale = init_std_scale
        self.inc = DoubleConv(in_channels, 64, is_first_block=True)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

        # 先用较小的随机权重稳定初始化网络，
        # 再把所有需要非负的卷积核和最后一层 bias 投影到非负区域，
        # 从而让 iNETT 的凸 U-Net 在训练开始前就满足凸性约束。
        self._initialize_weights()
        project_nonnegative_(self)

    def phi(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)

    def forward(self, x):
        # 训练 Phi_c 时使用
        return self.phi(x)

    def regularizer(self, x):
        # iNETT 迭代时使用
        phi_x = self.phi(x)
        dims = tuple(range(1, x.ndim))
        return self.a * torch.sum(x ** 2, dim=dims) + torch.sum(phi_x ** 2, dim=dims)

    def regularizer_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Return the batch-mean regularizer value."""
        return self.regularizer(x).mean()

    def regularizer_grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of mean R(x) with respect to x.

        This helper is useful when embedding the trained UCNN into an iterative
        reconstruction algorithm.
        """
        x = x.requires_grad_(True)
        r_value = self.regularizer_mean(x)
        grad = torch.autograd.grad(r_value, x, create_graph=True)[0]
        return grad

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                fan = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                # Positive-weight ICNN/UCNN structures can easily amplify activations.
                # Use a much smaller initialization than standard He initialization.
                std = self.init_std_scale * np.sqrt(2.0 / fan)
                nn.init.trunc_normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class CustomDataset(Dataset):
    """
    A custom dataset class to load data from dataset.npz file for UNet training.
    
    The dataset should contain arrays named 'xk', 'ydelta', 'xkdelta', and 'sk'.
    The model takes two inputs, so we need to decide how to combine these for training.
    
    专门为一维物理向量 (16384, 1) 设计的 U-Net 数据集加载器。
    它会自动将一维向量重塑为 PyTorch Conv2d 需要的 (Channels, Height, Width) 即 (1, 128, 128) 格式。
    """
    
    def __init__(self, npz_file_path, img_size=128):
        """
        Initialize the dataset by loading data from npz file.
        
        Args:
            npz_file_path (str): Path to the .npz file containing the data
        """
        # Load data from npz file
        data = np.load(npz_file_path)
        
        # Extract the required arrays
        self.xk = data['xk']
        self.xkdelta = data['xkdelta']
        
        self.length = len(self.xk)
        self.img_size = img_size
        
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.length
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset at index idx.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (xk, ydelta, xkdelta, sk)
        """
        # Get the data at the specified index
        xk_sample = self.xk[idx]
        xkdelta_sample = self.xkdelta[idx]
        
        # 转换为 PyTorch FloatTensor (使用 as_tensor 避免不必要的内存复制)
        xk_tensor = torch.as_tensor(xk_sample, dtype=torch.float32)
        xkdelta_tensor = torch.as_tensor(xkdelta_sample, dtype=torch.float32)

        # 步骤 A：先用 .view(-1) 抹平可能存在的 (16384, 1) 的尾部多余维度，变成纯粹的 (16384,)
        # 步骤 B：用 .view(1, img_size, img_size) 重塑为 (通道数, 高, 宽) -> (1, 128, 128)
        xk_tensor = xk_tensor.view(-1).view(1, self.img_size, self.img_size)
        xkdelta_tensor = xkdelta_tensor.view(-1).view(1, self.img_size, self.img_size)
        
        # Return the pair of inputs and the target
        return xk_tensor, xkdelta_tensor


def get_model(training_params, device, input_channels=1, n_classes=1):
    """
    获取模型、优化器和损失函数
    """
    # 初始化模型
    model = UNetinett(
        in_channels=input_channels,
        out_channels=n_classes,
        bilinear=training_params.get('bilinear', True)
    )
    
    # 移动模型到指定设备
    model = model.to(device)
    project_nonnegative_(model)
    
    # 定义优化器
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=training_params.get('learning_rate', 1e-3),
        weight_decay=training_params.get('weight_decay', 0)
    )
    
    # 定义损失函数
    if training_params.get('loss_fn') == 'MSE':
        error = nn.MSELoss()
    elif training_params.get('loss_fn') == 'L1':
        error = nn.L1Loss()
    elif training_params.get('loss_fn') == 'SmoothL1':
        error = nn.SmoothL1Loss()
    else:
        error = nn.MSELoss()  # 默认使用MSE
    
    return model, optimizer, error


def plot_losses(train_losses, test_losses, save_path):
    """
    绘制训练和测试损失曲线
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, test_losses, label='Test Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Curves')
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
def train(config):
    """
    训练函数
    """
    best_test_loss = float("inf")
    best_model_params = None
    best_epoch = 0
    
    # 获取模型、优化器和损失函数
    model, optimizer, error = get_model(
        config.training_params, 
        config.device,
        config.input_channels,
        config.n_classes
    )
    
    # 获取数据加载器
    TrainDataGen, TestDataGen = get_data_loader(config)
    
    # 清空GPU缓存
    torch.cuda.empty_cache()

    train_loss_list = np.zeros(config.epochs)
    test_loss_list = np.zeros(config.epochs)

    print("Starting training")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 每个epoch后评估模型性能并保存最佳模型
    for epoch in range(1, config.epochs + 1):
        train_loss = 0.0
        test_loss = 0.0

        model.train()
        # 训练阶段
        for batch_idx, (xk_batch, xkdelta_batch) in enumerate(TrainDataGen):
            # 将数据移到指定设备
            xk_batch = xk_batch.to(device=config.device, dtype=torch.float)
            xkdelta_batch = xkdelta_batch.to(device=config.device, dtype=torch.float)

            output = model(xkdelta_batch)
            target = torch.abs(xk_batch-xkdelta_batch)

            loss = error(output, target)
            
            # 反向传播
            optimizer.zero_grad()  # 清除之前的梯度
            loss.backward()        # 反向传播计算梯度
            optimizer.step()       # 更新模型参数
            project_nonnegative_(model)

            train_loss += loss.item()
            
            if batch_idx % 10 == 0:  # 每10个batch打印一次进度
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')

        # 验证阶段
        with torch.no_grad():
            model.eval()
            for xk_batch, xkdelta_batch in TestDataGen:
                xk_batch = xk_batch.to(device=config.device, dtype=torch.float)
                xkdelta_batch = xkdelta_batch.to(device=config.device, dtype=torch.float)
                
                output = model(xkdelta_batch)
                target = torch.abs(xk_batch-xkdelta_batch)
                loss = error(output, target)
                
                test_loss += loss.item()

        # 计算平均损失
        train_loss = train_loss / len(TrainDataGen)
        test_loss = test_loss / len(TestDataGen)

        # 保存最佳模型
        if test_loss < best_test_loss:
            best_epoch = epoch
            best_test_loss = test_loss
            best_model_params = {key: value.clone() for key, value in model.state_dict().items()}
            
            # 保存最佳模型
            model_save_path = "models/iNETT_best.pth"
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(best_model_params, model_save_path)
            print(f"New best model saved at epoch {epoch} with test loss: {test_loss:.8f}")

        print(
            f"Epoch [{epoch}/{config.epochs}], Train Loss: {train_loss:.8f}, "
            f"Test Loss: {test_loss:.8f}"
        )

        train_loss_list[epoch - 1] = train_loss
        test_loss_list[epoch - 1] = test_loss
        
        # 绘制损失曲线
        if epoch % config.plot_freq == 0 or epoch == 1:  # 每隔几个epoch绘制一次或第一个epoch
            plot_losses(
                train_loss_list[:epoch],
                test_loss_list[:epoch],
                f"losses/iNETT_loss_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )

    print("Training completed")
    print(f"Best epoch: {best_epoch}, Best test loss: {best_test_loss:.8f}")
    
    # 最终保存最佳模型
    final_model_path = "models/iNETT_final_best.pth"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(best_model_params, final_model_path)
    print(f"Final best model saved to {final_model_path}")

    print("=" * 60)


# 配置类
class Config:
    def __init__(self):
        # 数据集配置
        # self.dataset_path = "data/NETT_dataset.npz"  # 您的数据文件路径
        self.train_dataset_path = "data/train_iNETT_dataset.npz"  # 训练集文件路径
        self.val_dataset_path = "data/val_iNETT_dataset.npz" 
        self.input_channels = 1  # xkdelta的通道数
        self.n_classes = 1        # 输出通道数 (xkdelta)
        self.batch_size = 8
        self.train_ratio = 0.8
        self.num_workers = 4
        self.image_size = 128
        
        # 训练配置
        self.epochs = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 其他配置
        self.plot_freq = 10  # 每5个epoch绘制一次损失曲线
        
        # 训练参数
        self.training_params = {
            'learning_rate': 5e-5,
            'weight_decay': 1e-5,
            'bilinear': True,
            'loss_fn': 'SmoothL1'   # 可选: 'MSE', 'L1', 'SmoothL1'
        }


if __name__ == "__main__":
    # 创建配置
    config = Config()

    # 开启 cuDNN 基准测试，网络结构不变时大幅提升卷积性能
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # 检查数据文件是否存在
    if not os.path.exists(config.train_dataset_path):
        raise FileNotFoundError(f"Dataset file {config.train_dataset_path} not found!")
    if not os.path.exists(config.val_dataset_path):
        raise FileNotFoundError(f"Dataset file {config.val_dataset_path} not found!")
    
    print("Training configuration:")
    print(f"- Train Dataset: {config.train_dataset_path}")
    print(f"- Val Dataset: {config.val_dataset_path}")
    print(f"- Device: {config.device}")
    print(f"- Epochs: {config.epochs}")
    print(f"- Batch size: {config.batch_size}")
    print(f"- Learning rate: {config.training_params['learning_rate']}")
    print("- Model: UNet with dual inputs (xkdelta) -> output (xk)")
    print("=" * 60)
    
    # 创建保存目录
    os.makedirs("models", exist_ok=True)
    os.makedirs("losses", exist_ok=True)

    train(config=config)
    
    print("All models trained successfully!")
