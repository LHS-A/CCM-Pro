# -- coding: utf-8 --
import torch
import numpy as np
from metric import batch_metrics_pred
from utils import calculate_mean_and_std
from Loss_utils import Compute_KDloss

def val_S(args,val_loader,S_model,T_model,criterion):  
    print("================================== Valid epoch {} student model ====================================".format(args.epoch_S))
    epoch_loss,batch_sen_pred,batch_dice_pred,batch_pre_pred,batch_fdr_pred,batch_IOU_pred= [],[],[],[],[],[]
    S_model.eval()
    T_model.eval()

    with torch.no_grad(): 
        for image_lst,image_dia_lst,label_lst,label_dia_lst,res_label_dia_lst,image_name_lst in val_loader:
            image_dia = image_dia_lst.float().to(args.device)
            image = image_lst.float().to(args.device)
            label = label_lst.float().to(args.device)

            with torch.no_grad():
                T_label = T_model(image_dia)
                Feas_teacher = T_model.feas 
                
            pred = S_model(image)
            Feas_student = S_model.feas

            pred_BCE = criterion["BCEloss"](pred, label) #about possibility,all inputs are the same shape!
            pred_dice = criterion["DiceLoss"](pred, label)
            KD_loss = Compute_KDloss(Feas_teacher,Feas_student,label)
            loss = 0.5 * pred_BCE + 0.5* pred_dice + args.KD_cof * KD_loss
                
            epoch_loss.append(loss.item())

            pred_sen, pred_dice, pred_pre, pred_FDR, pred_IOU = batch_metrics_pred(image,torch.sigmoid(pred),label, image_name_lst, args.val_batch)
            batch_sen_pred.append(pred_sen);batch_dice_pred.append(pred_dice);batch_pre_pred.append(pred_pre);batch_fdr_pred.append(pred_FDR);batch_IOU_pred.append(pred_IOU)
        
        sen_pred, sen_pred_std, sen_percls_mean,sen_percls_std = calculate_mean_and_std(batch_sen_pred)
        dice_pred, dice_pred_std, dice_percls_mean,dice_percls_std = calculate_mean_and_std(batch_dice_pred)
        pre_pred, pre_pred_std, pre_percls_mean,pre_percls_std = calculate_mean_and_std(batch_pre_pred)
        fdr_pred, fdr_pred_std, fdr_percls_mean,fdr_percls_std = calculate_mean_and_std(batch_fdr_pred)
        IOU_pred, IOU_pred_std, IOU_percls_mean,IOU_percls_std = calculate_mean_and_std(batch_IOU_pred)
        
        print("================================ Epoch:{} Student Valid Metric =====================================".format(args.epoch_S))
        print("sen_PerCls: {}±{}, dice_PerCls: {}±{}, pre_PerCls: {}±{}, fdr_PerCls: {}±{}, IOU_PerCls: {}±{}".format(sen_percls_mean,sen_percls_std,dice_percls_mean,dice_percls_std,pre_percls_mean,pre_percls_std,fdr_percls_mean,fdr_percls_std,IOU_percls_mean,IOU_percls_std))
        
        return np.mean(epoch_loss),dice_pred
       