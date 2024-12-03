# -- coding: utf-8 --
import os
import numpy as np
import torch
import torch.nn.functional as F

def check_overfitting_T(optimizer,args):
    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
    if current_lr != args.init_lr_T:
        args.init_lr_T = current_lr
        args.mark_T_epoch = args.epoch_T
        args.train_teacher = False
     
def delete_previous_models(folder_path):
    files = os.listdir(folder_path)
    model_files = [file for file in files if file.endswith('.pkl') or file.endswith('.pth')]
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))
    if len(model_files) > 6:
        file_to_delete = model_files[0]
        file_path = os.path.join(folder_path, file_to_delete)
        os.remove(file_path)

def calculate_mean_and_std(metric):
    """
    Input: list; 
    Function: calculate metrics.
    """
    metric = np.array(metric)
    # Per class metric of mean±std.
    mean_metric = np.mean(metric,axis=0)
    std_metric = np.std(metric,axis=0)
    # Total metric of mean±std.
    mean_total = np.mean(mean_metric,axis=0)
    std_total = np.std(std_metric,axis=0)

    return mean_total,std_total,mean_metric, std_metric

def min_max_normalize(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())

def downsample_upsample_labels(label):
    # 确保label是一个PyTorch张量
    device = label.device  
    H,W = label.shape[-2:]
    # 使用F.interpolate进行缩放
    scale192 = F.interpolate(label.unsqueeze(0), size=(H // 2, H // 2), mode='bilinear', align_corners=True).squeeze(0)
    scale192 = torch.where(scale192 > 0.5, torch.tensor(1.0, dtype=torch.uint8).to(device), torch.tensor(0.0, dtype=torch.uint8).to(device))

    scale96 = F.interpolate(label.unsqueeze(0), size=(H // 4, H // 4), mode='bilinear', align_corners=True).squeeze(0)
    scale96 = torch.where(scale96 > 0.5, torch.tensor(1.0, dtype=torch.uint8).to(device), torch.tensor(0.0, dtype=torch.uint8).to(device))

    scale48 = F.interpolate(label.unsqueeze(0), size=(H // 8, H // 8), mode='bilinear', align_corners=True).squeeze(0)
    scale48 = torch.where(scale48 > 0.5, torch.tensor(1.0, dtype=torch.uint8).to(device), torch.tensor(0.0, dtype=torch.uint8).to(device))

    scale24 = F.interpolate(label.unsqueeze(0), size=(H // 16, H // 16), mode='bilinear', align_corners=True).squeeze(0)
    scale24 = torch.where(scale24 > 0.5, torch.tensor(1.0, dtype=torch.uint8).to(device), torch.tensor(0.0, dtype=torch.uint8).to(device))

    # 创建一个包含所有缩放标签的列表
    scales = [label, scale192, scale96, scale48, scale24, scale48, scale96, scale192, label, label]
 
    return scales
