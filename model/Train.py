import os
import torch
import timeit
import random
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from Evaluation import TSS_HSS
from torch.optim import lr_scheduler
from torch.cuda.amp import autocast, GradScaler
from torchvision.models.inception import InceptionOutputs

def seed_everything(seed: int):
    """固定所有随机种子保证结果可复现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """执行一个epoch的训练"""
    model.train()
    total_loss = 0.0
    y_all, preds_all = [], []

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        # 使用混合精度训练
        with autocast(enabled=scaler is not None):
            outputs = model(X)
            # 若为多输出，则取logits
            if type(outputs) == InceptionOutputs: outputs = outputs.logits 
            loss = criterion(outputs, y)    
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        y_all.append(y)
        preds_all.append(preds)

    # 拼接所有batch结果
    y_all = torch.cat(y_all)
    preds_all = torch.cat(preds_all)
    return total_loss / len(dataloader), preds_all, y_all

def evaluate(model, dataloader, criterion, device):
    """进行模型评估"""
    model.eval()
    total_loss = 0.0
    y_all, preds_all = [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            y_all.append(y)
            preds_all.append(preds)
    
    # 拼接所有batch结果
    y_all = torch.cat(y_all)
    preds_all = torch.cat(preds_all)
    return total_loss / len(dataloader), preds_all, y_all

def train(
    model,                      # 模型
    train_iter,                 # 训练集
    val_iter,                   # 验证集
    test_iter,                  # 测试集
    num_epochs,                 # 训练次数
    lr,                         # 学习率
    weight_decay,               # 学习率衰减
    device,                     # 设备
    saving_directory,           # 保存路径
    lr_decay=True,              # 学习率衰减      
    lr_decay_step=5,            # 衰减周期
    use_amp=False,              # 混合精度
    criterion_weight=None,      # 损失函数权重
    test_index = None,          # 测试集索引
    best_X_val_TSS_limit=0.0    # 最佳X_val_TSS阈值
):
    
    seed_everything(42)
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore")

    model_name = model._get_name()

    model.to(device)
    print('training on', device)
    
    # 初始化训练组件
    criterion = nn.CrossEntropyLoss(criterion_weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=0.5) if lr_decay else None
    scaler = GradScaler(enabled=use_amp) if device == 'cuda' and use_amp else None

    # 初始化结果字典
    results = results_dict()

    # 初始化最佳验证TSS
    best_val_TSS = 0.

    # 初始化测试保存路径
    test_save_path = os.path.join(saving_directory, "test_save")
    os.makedirs(test_save_path, exist_ok=True)

    # 开始训练循环
    for epoch in range(1, num_epochs+1):
        # ----------------- 日志 -----------------
        log_file = open(f"{saving_directory}/{model_name}_log.txt", mode='a')

        # ----------------- 训练 -----------------
        start_time = timeit.default_timer()
        train_loss, train_preds, train_y = train_epoch(model, train_iter, criterion, optimizer, device, scaler)
        train_time = timeit.default_timer() - start_time

        # ----------------- 验证 -----------------
        start_time = timeit.default_timer()
        val_loss, val_preds, val_y = evaluate(model, val_iter, criterion, device)
        val_time = timeit.default_timer() - start_time

        # ----------------- 测试 -----------------
        start_time = timeit.default_timer()
        test_loss, test_preds, test_y = evaluate(model, test_iter, criterion, device)
        test_time = timeit.default_timer() - start_time
        
        # ----------------- 更新学习率 -----------------
        if scheduler: scheduler.step()

        # ----------------- 保存训练结果 -----------------
        current_lr = optimizer.param_groups[0]['lr']
        results['epoch'].append(epoch)
        results['learning_rate'].append(current_lr)
        results_save(results, train_time, train_loss, train_preds, train_y, 'train')
        results_save(results, val_time, val_loss, val_preds, val_y, 'val')
        results_save(results, test_time, test_loss, test_preds, test_y, 'test')

        # ----------------- 保存日志 -----------------
        log_write(results, log_file); log_file.close()

        test_save(test_preds, test_y, test_index, f"{test_save_path}/{epoch}.csv")

        # ----------------- 保存中间的最佳模型 -----------------
        M_val_TSS = results['M_val_TSS'][-1]
        X_val_TSS = results['X_val_TSS'][-1]
        if M_val_TSS > best_val_TSS and X_val_TSS > best_X_val_TSS_limit:
            best_val_TSS = M_val_TSS
            torch.save(model.state_dict(), f"{saving_directory}/{model_name}_best.pth")

    # 保存最终模型
    torch.save(model.state_dict(), f"{saving_directory}/{model_name}.pth")
    # 保存最终训练结果
    pd.DataFrame(results).to_csv(f"{saving_directory}/{model_name}_results.csv", index=False, header=True)

def results_dict():
    """初始化结果字典"""
    results = {'epoch': [], 'learning_rate': [],}

    phases = ['train', 'val', 'test']
    classes = ['C', 'M', 'X']
    metrics = ['TSS', 'HSS', 'TP', 'FP', 'FN', 'TN']

    for phase in phases:
        results[f'{phase}_time'] = []
        results[f'{phase}_loss'] = []
        for metric in metrics:
            for cls in classes:
                results[f'{cls}_{phase}_{metric}'] = []

    return results

def results_save(results, time, loss, preds, y, phase):
    """计算并保存评估结果"""
    classes = ['C', 'M', 'X']
    metrics = ['TSS', 'HSS', 'TP', 'FP', 'FN', 'TN']

    results[f'{phase}_time'].append(time)
    results[f'{phase}_loss'].append(loss)

    for cls in classes:
        TP, FP, FN, TN, TSS, HSS = TSS_HSS(preds, y, grade=cls)
        metric_values = (TSS, HSS, TP, FP, FN, TN)

        for metric, value in zip(metrics, metric_values):
            results[f'{cls}_{phase}_{metric}'].append(value)

def log_write(results, log_file):
    """写入日志"""    
    log_lines = [
        f"Epoch {results['epoch'][-1]}:",
        f"  Learning Rate: {results['learning_rate'][-1]}",
        f"  Times: \t Train: {results['train_time'][-1]:.1f}s \t| Val: {results['val_time'][-1]:.1f}s \t| Test: {results['test_time'][-1]:.1f}s",
        f"  Losses: \t Train: {results['train_loss'][-1]:.5f} \t| Val: {results['val_loss'][-1]:.5f} \t| Test: {results['test_loss'][-1]:.5f}",
        "  TSS Scores:",
        f"    Train:\t C: {results['C_train_TSS'][-1]:.5f} \t| M: {results['M_train_TSS'][-1]:.5f} \t| X: {results['X_train_TSS'][-1]:.5f}",
        f"    Val:\t C: {results['C_val_TSS'][-1]:.5f} \t| M: {results['M_val_TSS'][-1]:.5f} \t| X: {results['X_val_TSS'][-1]:.5f}",
        f"    Test:\t C: {results['C_test_TSS'][-1]:.5f} \t| M: {results['M_test_TSS'][-1]:.5f} \t| X: {results['X_test_TSS'][-1]:.5f}",
        "  HSS Scores:",
        f"    Train:\t C: {results['C_train_HSS'][-1]:.5f} \t| M: {results['M_train_HSS'][-1]:.5f} \t| X: {results['X_train_HSS'][-1]:.5f}",
        f"    Val:\t C: {results['C_val_HSS'][-1]:.5f} \t| M: {results['M_val_HSS'][-1]:.5f} \t| X: {results['X_val_HSS'][-1]:.5f}",
        f"    Test:\t C: {results['C_test_HSS'][-1]:.5f} \t| M: {results['M_test_HSS'][-1]:.5f} \t| X: {results['X_test_HSS'][-1]:.5f}"]
    
    log = '\n'.join(log_lines)
    print(log, file=log_file)

def test_save(test_preds, test_y, date, save_path):
    """保存测试集结果"""
    mapping = {0: 'NF', 1: 'C', 2: 'M', 3: 'X'}

    test_preds = [mapping.get(i) for i in test_preds.cpu().numpy()]
    test_y = [mapping.get(i) for i in test_y.cpu().numpy()]

    test_df = {'date': date,
               'label': test_y,
               'pred': test_preds}
    
    test_df = pd.DataFrame(test_df)
    test_df.to_csv(save_path, index=False, header=True)