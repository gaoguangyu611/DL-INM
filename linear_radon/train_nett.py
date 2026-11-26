from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split


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
class UNetNETT(nn.Module):
    

    def __init__(
        self,
        input_channels: int,
        n_classes: int,            # 输出通道数
        radon_full=None,
        bilinear: bool = True,
    ):

        super(UNetNETT, self).__init__()
        # 计算总输入通道数
        self.n_channels = input_channels
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


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        x1 = self.inc(x)
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


def get_data_loaders(dataset_path, batch_size, train_ratio, shuffle, num_workers, img_size=128):
    """
    Helper function to create train and validation data loaders from the custom dataset.
    
    Args:
        dataset_path (str): Path to the .npz file
        batch_size (int): Batch size for the data loaders
        train_ratio (float): Proportion of data to use for training (rest for validation)
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of worker processes for data loading
        img_size (int): Expected image height/width
        
    Returns:
        tuple: (train_loader, val_loader) PyTorch DataLoader objects
    """
    # Create the full dataset
    full_dataset = CustomDataset(dataset_path, img_size=img_size)

    # Calculate split sizes
    total_size = len(full_dataset)
    print(f"数据集总大小: {total_size}")

    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size
    print(f"划分情况 -> 训练集: {train_size}, 验证集: {val_size}")

    # Randomly split the dataset
    # 可以设置 generator 保证每次划分结果一致（可选）
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=generator
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True # 如果使用GPU加速，建议开启
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Usually don't shuffle validation data
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# 清空GPU缓存
torch.cuda.empty_cache()


def get_model(training_params, device, input_channels=1, n_classes=1):
    """
    获取模型、优化器和损失函数
    """
    # 初始化模型
    model = UNetNETT(
        input_channels=input_channels,
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
    获取训练和测试数据加载器
    """
    # 创建训练和测试数据加载器
    train_loader, val_loader = get_data_loaders(
        config.dataset_path,
        batch_size=config.batch_size,
        train_ratio=config.train_ratio,
        shuffle=True,
        num_workers=config.num_workers,
        img_size = config.image_size
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
            loss = error(output, xk_batch)
            
            # 反向传播
            optimizer.zero_grad()  # 清除之前的梯度
            loss.backward()        # 反向传播计算梯度
            optimizer.step()       # 更新模型参数

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
                loss = error(output, xk_batch)
                
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
            model_save_path = "models/NETT_best.pth"
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
                f"losses/NETT_loss_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )

    print("Training completed")
    print(f"Best epoch: {best_epoch}, Best test loss: {best_test_loss:.8f}")
    
    # 最终保存最佳模型
    final_model_path = "models/NETT_final_best.pth"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(best_model_params, final_model_path)
    print(f"Final best model saved to {final_model_path}")

    print("=" * 60)


# 配置类
class Config:
    def __init__(self):
        # 数据集配置
        self.dataset_path = "data/NETT_dataset.npz"  # 您的数据文件路径
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
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'bilinear': True,
            'loss_fn': 'MSE'   # 可选: 'MSE', 'L1', 'SmoothL1'
        }


if __name__ == "__main__":
    # 创建配置
    config = Config()

    # 开启 cuDNN 基准测试，网络结构不变时大幅提升卷积性能
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # 检查数据文件是否存在
    if not os.path.exists(config.dataset_path):
        raise FileNotFoundError(f"Dataset file {config.dataset_path} not found!")
    
    print("Training configuration:")
    print(f"- Dataset: {config.dataset_path}")
    print(f"- Device: {config.device}")
    print(f"- Epochs: {config.epochs}")
    print(f"- Batch size: {config.batch_size}")
    print(f"- Learning rate: {config.training_params['learning_rate']}")
    print("- Model: UNet with dual inputs (xkdelta) -> output (xk)")
    print("=" * 60)
    
    # 创建保存目录
    os.makedirs("models", exist_ok=True)
    os.makedirs("losses", exist_ok=True)

    if not os.path.exists(config.dataset_path):
        raise FileNotFoundError(f"Dataset file {config.dataset_path} not found!")
    
    # for model_name in model_names:
    #     train(
    #         example="dual_input_training",
    #         model_name=model_name,
    #         config=config,
    #     )

    train(config=config)
    
    print("All models trained successfully!")
