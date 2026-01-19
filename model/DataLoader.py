import os
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from torchsampler import ImbalancedDatasetSampler
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class MAG_Dataset(Dataset):
    """
    全日面磁场数据集
    """
    def __init__(self, csv, root, transform = None):
        """
        csv : 数据集csv文件
        root : 数据集存放目录
        transform : 数据增强
        """
        if isinstance(csv, str):
            self.csv = pd.read_csv(csv)
        if isinstance(csv, pd.DataFrame):
            self.csv = csv           

        self.csv['date'] = pd.to_datetime(self.csv['date']) 
        self.mapping = {'NF':0, 'C':1, 'M':2, 'X':3}        
        self.csv['label'] = self.csv['label'].map(self.mapping)

        self.root = root
        self.transform = transform

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        data_path = self.data_path(row['date'], row['file'])    # 文件路径
        data = np.load(data_path)['data'].astype(np.float32)    # 读取文件
        data = np.stack([data]*3, axis=-1)                      # 堆叠成三通道
        if self.transform: data = self.transform(data)          # 变换
        return (data, torch.tensor(row['label']))

    def __len__(self):
        return len(self.csv)
    
    def data_path(self, date, file):
        """生成文件路径"""
        return os.path.join(self.root, f"{date.year}/{date.month}/{date.day}", file)

    def get_labels(self):
        """返回数据集标签, 用于ImbalancedDatasetSampler"""
        return self.csv['label']

def load_mag_data(csv, root, transform, 
                  batch_size, shuffle, num_workers):
    """
    加载磁图数据集

    csv : 数据集csv文件
    root : 数据集根目录
    transform : 数据增强
    batch_size : 批次大小
    shuffle : 是否打乱数据
    num_workers : 读取数据的子进程数
    """
    mag_dataset = MAG_Dataset(csv, root, transform)
    loader = DataLoader(
        dataset = mag_dataset, 
        batch_size = batch_size, 
        shuffle = shuffle, 
        num_workers = num_workers, 
        pin_memory = True)
    return loader

def load_mag_data_augment(csv, root,
                          batch_size, shuffle, num_workers,
                          X_aug = True, M_aug = True):
    """
    加载磁图数据集, 并使用数据增强
    """
    # 基础变换
    base = transforms.Compose([
        transforms.ToTensor() ])
    
    # 数据增强1: 随机旋转
    rotation = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=(-5, 5))])
    
    # 数据增强2: 水平翻转
    hr_flip = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=1.0)])
    
    # 数据增强3：垂直翻转
    vr_flip = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomVerticalFlip(p=1.0)])
    
    # 数据增强4：随机选择一个数据增强
    random = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomChoice([
            transforms.RandomRotation(degrees=(-5, 5)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0)])])

    # 筛选出各等级数据
    csv = pd.read_csv(csv)
    X = csv[csv['label'] == 'X'].copy(); X.reset_index(drop=True, inplace=True)
    M = csv[csv['label'] == 'M'].copy(); M.reset_index(drop=True, inplace=True)
    C = csv[csv['label'] == 'C'].copy(); C.reset_index(drop=True, inplace=True)
    NF = csv[csv['label'] == 'NF'].copy(); NF.reset_index(drop=True, inplace=True)

    # 数据集列表
    dataset_list = []
    
    # X级数据增强
    if X_aug:
        X_rotation = MAG_Dataset(X.copy(), root, rotation)
        X_hr_flip = MAG_Dataset(X.copy(), root, hr_flip)
        X_vr_flip = MAG_Dataset(X.copy(), root, vr_flip)
        dataset_list.extend([X_rotation, X_hr_flip, X_vr_flip])

    # M级数据增强
    if M_aug:
        M_random = MAG_Dataset(M.copy(), root, random)
        dataset_list.append(M_random)

    # 基础变换
    X_base = MAG_Dataset(X.copy(), root, base)
    M_base = MAG_Dataset(M.copy(), root, base)
    C_base = MAG_Dataset(C.copy(), root, base)
    NF_base = MAG_Dataset(NF.copy(), root, base)
    dataset_list.extend([X_base, M_base, C_base, NF_base])

    # 根据数据集列表加载数据集
    loader = DataLoader(
        ConcatDataset(dataset_list),
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
        pin_memory = True)
    
    return loader

def load_mag_data_sampler(csv:str, root, transform, 
                          batch_size, num_workers, sampler_type = 'auto'):
    """
    加载磁图数据集, 并使用重采样器

    sampler_type:

        auto : 自动采样器ImbalancedDatasetSampler

        weighted : 加权采样器WeightedRandomSampler
        
    """
    mag_dataset = MAG_Dataset(csv, root, transform)
    
    if sampler_type == 'auto': 
        _sampler = ImbalancedDatasetSampler(mag_dataset)
        
    if sampler_type == 'weighted':
        csv_df = pd.read_csv(csv)
        sampler_dict = csv_df['label'].value_counts(normalize=True).to_dict()
        label_weight = csv_df['label'].apply(lambda x: 1 / sampler_dict.get(x))
        weight = torch.tensor(label_weight.to_numpy(), dtype=torch.float32)
        _sampler = WeightedRandomSampler(weight, num_samples = len(csv_df),replacement=True)

    loader = DataLoader(dataset = mag_dataset, 
                        sampler = _sampler,
                        batch_size = batch_size, 
                        num_workers = num_workers, 
                        pin_memory = True)
        
    return loader