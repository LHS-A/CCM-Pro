# -- coding: utf-8 --
import torch.nn as nn
import torch.optim as optim
import random
import torch
from copy import deepcopy
from utils import *
from Loss_utils import DiceLoss
from config import *
args = Params()
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from Training_Process.Train_S import train_S
from Training_Process.Train_T import train_T
from Training_Process.Val_S import val_S
from Training_Process.Val_T import val_T
from Training_Process.Test_S import test_S
from Training_Process.Test_T import test_T
from Training_Process.dataloader import Data_loader

# from Network.Unet.model_S_adapter import Net_S
# from Network.Unet.model_T_adapter import Net_T 
from Network.Unet.model_S import Net_S 
from Network.Unet.model_T import Net_T

def model_installization(model,pretrained_weight):
    model_pretrain_dict = torch.load(pretrained_weight)
    model_dict = model.state_dict()
    model_pretrain_dict = {k:v for k,v in model_pretrain_dict.items() if k in model_dict}  # get the same layer! if your model do not has pretrain module,then it will be random installization!
    model_dict.update(model_pretrain_dict) 
    model.load_state_dict(model_dict)

    for name,param in model.named_parameters():
        if name in model_pretrain_dict:   
            param.requires_grad = False
        else: 
            param.requires_grad = True

    return model

def model_settings(args):
    S_model = Net_S(args.input_dim,args.num_classes).to(args.device)
    T_model = Net_T(args.input_dim,args.num_classes).to(args.device)
    # S_model = model_installization(S_model,(r"/home/imed/personal/LHS/CCM-Pro/CNs_Proposed_best.pkl"))
    # T_model =  model_installization(T_model,(r"/home/imed/personal/LHS/CCM-Pro/CNs_Proposed_best.pkl"))

    return S_model,T_model

##========================================================== 模型参数设置 =========================================================
seed = args.seed_random
torch.manual_seed(seed) 
torch.cuda.manual_seed_all(seed) 
np.random.seed(seed) 
random.seed(seed)  

S_model, T_model = model_settings(args)
# T_model = torch.nn.DataParallel(T_model, device_ids = args.device_ids)
# S_model = torch.nn.DataParallel(S_model, device_ids = args.device_ids)

optimizer_S = optim.Adam(S_model.parameters(), lr = args.init_lr_S)
optimizer_T = optim.Adam(T_model.parameters(), lr = args.init_lr_T)
scheduler_T = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_T, mode="min", factor=0.95, patience=6, verbose=True, threshold=0, threshold_mode="rel", cooldown=0, min_lr=1e-6, eps=1e-08)
criterion  = {"BCEloss":nn.BCEWithLogitsLoss().to(args.device),"DiceLoss":DiceLoss(args.num_classes).to(args.device)}

data_loader = Data_loader()
val_loader = data_loader.load_val_data(args,batch_size = args.val_batch)
test_loader = data_loader.load_test_data(args,batch_size = args.test_batch)
train_loader = data_loader.load_train_data(args,batch_size = args.train_batch) 

if args.checkpoint == True:
    T_model.load_state_dict(torch.load(r"/home/imed/personal/LHS/CCM-Pro/model_weight/best_model/T/2025TMI_T_Stroma_NerveFormer_propose/best_epoch_19.pkl"))
    args.train_teacher = False
    args.epoch_T = 20 
    args.mark_T_epoch = 20
    args.best_epoch = 19

while args.count_KD < len(args.dialated_pixels_list): 
    if args.train_teacher == True: 
        print("Start training Teacher!")
        args.enhance_mode_T = None #"train"
 
        for epoch in range(args.epoch_T + 1,args.epochs_T): 
            args.epoch_T = epoch  
            train_T(args,train_loader,T_model,optimizer_T,criterion)
            loss_T,dice_T = val_T(args,val_loader,T_model,criterion)
 
            if dice_T > args.best_dice_T:
                args.best_dice_T = dice_T 
                torch.save(T_model.state_dict(), args.T_Bestmodel_path  + "/best_epoch_{}.pkl".format(args.epoch_T))
                args.best_epoch = args.epoch_T 
                delete_previous_models(args.T_Bestmodel_path)  
 
            if args.epoch_T - args.mark_T_epoch == args.teacher_thed:
                args.mark_T_epoch = args.epoch_T  
                args.train_teacher = False
            else: 
                scheduler_T.step(loss_T)
                check_overfitting_T(optimizer_T,args)

            if args.train_teacher == False:
                if dice_T <= args.best_dice_T:    
                    print("Installize the best epoch:{} teacher model.".format(args.best_epoch))
                    T_model.load_state_dict(torch.load(args.T_Bestmodel_path  + "/best_epoch_{}.pkl".format(args.best_epoch))) 
                break  
                    
    else:
        print("Start training Student!")
        args.enhance_mode_S = "train"
        
        for epoch in range(args.epoch_S + 1,args.epochs_S):
            args.epoch_S = epoch  
            if args.count_KD == 0:
                args.KD_cof = min(args.epoch_S / args.value_thed,0.8)
                # args.KD_cof = 0
            else:  
                args.KD_cof = 0.8
            
            train_S(args,train_loader,S_model,T_model,optimizer_S,criterion)
            loss_S,dice_pred = val_S(args,val_loader,S_model,T_model,criterion) 
        
            if dice_pred > args.best_dice_S:
                args.best_dice_S = dice_pred
                torch.save(S_model.state_dict(), args.S_Bestmodel_path + "/best_epoch_{}.pkl".format(args.epoch_S))
                delete_previous_models(args.S_Bestmodel_path)  

            if args.epoch_S - args.mark_S_epoch == args.value_thed: 
                args.train_teacher = True  
                args.mark_S_epoch = args.epoch_S
                args.count_KD += 1 

            if args.train_teacher == True: 
                print("Stage:{} is finished!".format(args.count_KD))
                break     

         
          
            

                
            
    
