"""Historical inference architecture and four-field test dataset adapter."""
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
        self.ydelta = data['ydelta']
        self.xkdelta = data['xkdelta']
        self.sk = data['sk']
        
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
        ydelta_sample = self.ydelta[idx]
        xkdelta_sample = self.xkdelta[idx]
        sk_sample = self.sk[idx]

        xk_tensor = torch.as_tensor(xk_sample, dtype=torch.float32)
        ydelta_tensor = torch.as_tensor(ydelta_sample, dtype=torch.float32)
        xkdelta_tensor = torch.as_tensor(xkdelta_sample, dtype=torch.float32)
        sk_tensor = torch.as_tensor(sk_sample, dtype=torch.float32)

        xk_tensor = xk_tensor.view(-1).view(1, self.img_size, self.img_size)
        xkdelta_tensor = xkdelta_tensor.view(-1).view(1, self.img_size, self.img_size)
        sk_tensor = sk_tensor.view(-1).view(1, self.img_size, self.img_size)
        
        # ydelta 是投影数据（Sinogram），通常大小是 (7680,) 即 60角度x128探测器。
        ydelta_tensor = ydelta_tensor.view(-1)
        
        # Return the pair of inputs and the target
        return xk_tensor, ydelta_tensor, xkdelta_tensor, sk_tensor
