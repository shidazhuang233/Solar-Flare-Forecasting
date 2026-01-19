import os
import torch
import random
import numpy as np
import torch.nn as nn
from torch import Tensor
from Initialization import *
from torchvision import models

def seed_everything(seed: int):
    """固定所有随机种子保证结果可复现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class BaseModel(nn.Module):
    def __init__(self):
        """基础类, 用于模型参数初始化"""
        super().__init__()
    
    @staticmethod
    def _apply_initialization(module, init):
        seed_everything(42)
        if init == 'kaimingNormal':
            weights_init_kaimingNormal(module)
        elif init == 'kaimingUniform':
            weights_init_kaimingUniform(module)
        elif init == 'xavierNormal':
            weights_init_xavierNormal(module)
        elif init == 'xavierUniform':
            weights_init_xavierUniform(module)
        else:
            raise NotImplementedError("Invalid type of initialization!")

class AlexNet(BaseModel):
    def __init__(self, num_classes = 4, init='kaimingNormal'):
        super().__init__()

        # AlexNet网络的特征提取部分（加入BatchNorm）
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(192), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384), 
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
            )

        # 全局池化层
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # 重写分类器（加入BatchNorm）
        self.classifier = nn.Sequential(
            nn.Dropout(), 
            nn.Linear(256*6*6, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True), 
            nn.Dropout(), 
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes))

        # 初始化全连接层的权重
        self._apply_initialization(self, init)
        
    def forward(self, x:Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class VGG(BaseModel):
    def __init__(self, num_classes = 4, init='kaimingNormal'):
        super().__init__()
        
        # VGG网络的特征提取部分（加入BatchNorm）
        self.features = nn.Sequential(
            BasicConv(3, 64, kernel_size=3, stride=1, padding=1),
            BasicConv(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BasicConv(64, 128, kernel_size=3, stride=1, padding=1),
            BasicConv(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BasicConv(128, 256, kernel_size=3, stride=1, padding=1),
            BasicConv(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BasicConv(256, 512, kernel_size=3, stride=1, padding=1),
            BasicConv(512, 512, kernel_size=3, stride=1, padding=1), 
            BasicConv(512, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BasicConv(512, 512, kernel_size=3, stride=1, padding=1),
            BasicConv(512, 512, kernel_size=3, stride=1, padding=1),
            BasicConv(512, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 全局池化层
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # 重写分类器（加入BatchNorm）
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096), 
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True), 
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True), 
            nn.Dropout(),
            nn.Linear(4096, num_classes))

        self._apply_initialization(self, init)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def Inception3(num_classes=4, weights="Inception_V3_Weights.DEFAULT"):
    model = models.inception_v3(weights= weights)
    model.fc = nn.Linear(in_features=2048, out_features=num_classes)
    return model

def ResNet(num_classes=4, weights="ResNet34_Weights.DEFAULT"):
    model = models.resnet34(weights = weights)
    model.fc = nn.Linear(in_features=512, out_features=num_classes)
    return model

def VisionTransformer(num_classes=4, weights="ViT_B_16_Weights.DEFAULT"):
    model = models.vit_b_16(weights = weights)
    model.heads.head = nn.Linear(in_features=768, out_features=num_classes)
    return model