# -- coding: utf-8 --
import numpy as np
import cv2
from Loss_utils import iou
import warnings
warnings.filterwarnings("ignore")

def single_categories_count(pred, gt):

    FP = float(np.sum((pred == 255) & (gt != 255)))
    FN = float(np.sum((pred != 255) & (gt == 255)))
    TP = float(np.sum((pred == 255) & (gt == 255)))
    TN = float(np.sum((pred != 255) & (gt != 255)))

    return FP, FN, TP, TN

def get_single_indicator(pred, gt):
    FP, FN, TP, TN = single_categories_count(pred, gt)
    if pred.sum() > 0 and gt.sum() > 0:          
        sen = (TP) / (TP + FN + 1e-10)
        dice = (2 * TP) / ((TP + FN) + (TP + FP) + 1e-10)
        FDR = FP / (FP + TP + 1e-10)
        pre = TP / (TP + FP + 1e-10)
        IOU = iou(pred/255.0,gt/255.0)
 
        return sen, dice, IOU, FDR, pre

    elif pred.sum() == 0 and gt.sum() == 0: 
        return 1, 1, 1, 0, 0
    else:
        return 0, 0, 0, 1, 1

def batch_metrics_pred(image_batch ,pred_batch, pred_label_batch,img_name,num_batch):
    sen_batch, dice_batch, MHD_batch, FDR_batch, IOU_batch=  [],[],[],[],[]
    pred_batch = (pred_batch.detach().cpu().numpy() * 255).astype(np.uint8)
    pred_label_batch = (pred_label_batch.detach().cpu().numpy() * 255).astype(np.uint8)
    image_batch = (image_batch.detach().cpu().numpy() * 255).astype(np.uint8)
    for j in range(pred_batch.shape[1]):
        sen_batch.append([]);dice_batch.append([]);MHD_batch.append([]);FDR_batch.append([]);IOU_batch.append([])  
    for i in range(num_batch):
        image = image_batch[i, :, :, :]
        pred_multi = pred_batch[i, :, :, :]
        pred_label_multi = pred_label_batch[i, :, :, :]
        name = img_name[i].split(".")[0]
        slices = []
        for j in range(pred_batch.shape[1]):
            _, pred = cv2.threshold(pred_multi[j,:,:], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            slices.append(pred)

            sen, dice, IOU, FDR, MHD = get_single_indicator(pred, pred_label_multi[j,:,:])
            sen_batch[j].append(sen);dice_batch[j].append(dice);MHD_batch[j].append(MHD);FDR_batch[j].append(FDR);IOU_batch[j].append(IOU)
    
        pred = np.stack(slices,axis=0)

    sen_batch = np.nanmean(sen_batch, axis=1);dice_batch = np.nanmean(dice_batch, axis=1)
    MHD_batch = np.nanmean(MHD_batch, axis=1);FDR_batch = np.nanmean(FDR_batch, axis=1)
    IOU_batch = np.nanmean(IOU_batch, axis=1)
  
    return sen_batch, dice_batch, IOU_batch, FDR_batch, MHD_batch 


