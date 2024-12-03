# -- coding: utf-8 --
import torch
from metric import batch_metrics_pred
from utils import calculate_mean_and_std
from copy import deepcopy
from visualizer import Visualizer
import numpy as np

def val_T(args,val_loader,T_model,criterion):
    print("================================== Valid epoch {} teacher model ====================================".format(args.epoch_T))
    epoch_loss,batch_sen_pred,batch_dice_pred,batch_pred_MHD,batch_fdr_pred,batch_IOU_pred = [],[],[],[],[],[]
    T_model.eval()
    with torch.no_grad():
        for image_lst,image_dia_lst,label_lst,label_dia_lst,res_label_dia_lst,image_name_lst in val_loader:
            image_dia = image_dia_lst.float().to(args.device)
            label = label_lst.float().to(args.device)
            
            pred = T_model(image_dia)

            pred_BCE = criterion["BCEloss"](pred, label)
            pred_dice = criterion["DiceLoss"](pred, label)

            loss = 0.5 * pred_BCE + 0.5* pred_dice 
            epoch_loss.append(loss.item()) 

            pred_sen, pred_dice, pred_IOU, pred_FDR, pred_MHD = batch_metrics_pred(image_dia,torch.sigmoid(pred),label, image_name_lst, args.test_batch)
            batch_sen_pred.append(pred_sen);batch_dice_pred.append(pred_dice);batch_pred_MHD.append(pred_MHD);batch_fdr_pred.append(pred_FDR);batch_IOU_pred.append(pred_IOU)
        
        sen_pred, sen_pred_std, sen_percls_mean,sen_percls_std = calculate_mean_and_std(batch_sen_pred)
        dice_pred, dice_pred_std, dice_percls_mean,dice_percls_std = calculate_mean_and_std(batch_dice_pred)
        pre_pred, pre_pred_std, pre_percls_mean,pre_percls_std = calculate_mean_and_std(batch_pred_MHD)
        fdr_pred, fdr_pred_std, fdr_percls_mean,fdr_percls_std = calculate_mean_and_std(batch_fdr_pred)
        IOU_pred, IOU_pred_std, IOU_percls_mean,IOU_percls_std = calculate_mean_and_std(batch_IOU_pred)
        
        print("================================ Epoch:{} Teacher Valid Metric =====================================".format(args.epoch_T))
        print("sen_PerCls: {}±{}, dice_PerCls: {}±{}, MHD_PerCls: {}±{}, fdr_PerCls: {}±{}, IOU_PerCls: {}±{}".format(sen_percls_mean,sen_percls_std,dice_percls_mean,dice_percls_std,pre_percls_mean,pre_percls_std,fdr_percls_mean,fdr_percls_std,IOU_percls_mean,IOU_percls_std))
          
        return np.mean(epoch_loss),dice_pred
      