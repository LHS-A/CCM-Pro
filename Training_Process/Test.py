# -- coding: utf-8 --

from utils import calculate_mean_and_std
from metric import batch_metrics_pred
from config import *
args = Params() 
from Training_Process.dataloader import Data_loader
# from Network.Unet.model_S_adapter import Net_S
from Network.Unet.model_S import Net_S 


def test(args,test_loader,model):
    batch_sen_pred,batch_dice_pred,batch_pre_pred,batch_fdr_pred,batch_IOU_pred= [],[],[],[],[]
    model.eval()
    with torch.no_grad():
        for image_lst,image_dia_lst,label_lst,label_dia_lst,res_label_dia_lst,image_name_lst in test_loader:
            image = image_lst.float().to(args.device)
            label = label_lst.float().to(args.device)
        
            pred = model(image)
        
            pred_sen, pred_dice, pred_pre, pred_FDR, pred_IOU = batch_metrics_pred(image,torch.sigmoid(pred),label,image_name_lst, args.test_batch)
            batch_sen_pred.append(pred_sen);batch_dice_pred.append(pred_dice);batch_pre_pred.append(pred_pre);batch_fdr_pred.append(pred_FDR);batch_IOU_pred.append(pred_IOU)
        
        sen_pred, sen_pred_std, sen_percls_mean,sen_percls_std = calculate_mean_and_std(batch_sen_pred)
        dice_pred, dice_pred_std, dice_percls_mean,dice_percls_std = calculate_mean_and_std(batch_dice_pred)
        pre_pred, pre_pred_std, pre_percls_mean,pre_percls_std = calculate_mean_and_std(batch_pre_pred)
        fdr_pred, fdr_pred_std, fdr_percls_mean,fdr_percls_std = calculate_mean_and_std(batch_fdr_pred)
        IOU_pred, IOU_pred_std, IOU_percls_mean,IOU_percls_std = calculate_mean_and_std(batch_IOU_pred)
                
        print("sen_PerCls: {}±{}, dice_PerCls: {}±{}, MHD_PerCls: {}±{}, fdr_PerCls: {}±{}, IOU_PerCls: {}±{}".format(sen_percls_mean,sen_percls_std,dice_percls_mean,dice_percls_std,pre_percls_mean,pre_percls_std,fdr_percls_mean,fdr_percls_std,IOU_percls_mean,IOU_percls_std))


if __name__ == "__main__":
    args.data_path = r"/home/imed/personal/LHS/CCM-Pro_public/Dataset/CORN_3"
    data_loader = Data_loader()
    test_loader = data_loader.load_test_data(args,batch_size = args.test_batch)

    model = Net_T(3,1).to(device)
    model.load_state_dict(torch.load(args.S_Bestmodel_path  + "/best_epoch_300.pkl")) 

    test(args,test_loader,model)

