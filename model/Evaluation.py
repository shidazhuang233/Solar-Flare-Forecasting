import torch
from sklearn.metrics import confusion_matrix

def TSS_HSS(
    preds: torch.Tensor | list[torch.Tensor],
    targets: torch.Tensor | list[torch.Tensor],
    grade: str = 'C', 
    pos_label: str = 'FL',
    neg_label: str = 'NF',
):
    """
    计算二分类的TSS和HSS
    """

    if grade == 'C':
        mapping = {0: 'NF', 1: 'FL', 2: 'FL', 3: 'FL'}
    if grade == 'M':
        mapping = {0: 'NF', 1: 'NF', 2: 'FL', 3: 'FL'}
    if grade == 'X':
        mapping = {0: 'NF', 1: 'NF', 2: 'NF', 3: 'FL'}

    # 统一输入格式处理
    if isinstance(preds, list): preds = torch.cat(preds)
    if isinstance(targets, list): targets = torch.cat(targets)

    # 转换为CPU numpy数组处理
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()

    # 应用分类映射
    pred_labels = [mapping.get(int(x), None) for x in preds]
    true_labels = [mapping.get(int(x), None) for x in targets]

    # 计算混淆矩阵
    cm = confusion_matrix(
        y_true=true_labels,
        y_pred=pred_labels,
        labels=[pos_label, neg_label]
    )

    # 解析混淆矩阵元素
    TP, FN, FP, TN = cm.ravel()

    # 计算TSS评分
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    TSS = sensitivity - (1 - specificity)  # 等效于TPR - FPR
    
    # 计算HSS评分
    denominator = (TP + FN)*(FN + TN) + (TP + FP)*(FP + TN)
    if denominator == 0:
        HSS = 0.0
    else:
        HSS = 2 * (TP*TN - FN*FP) / denominator

    return TP, FP, FN, TN, TSS, HSS