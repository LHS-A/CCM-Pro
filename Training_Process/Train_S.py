# -- coding: utf-8 --
import torch
from Loss_utils import Compute_KDloss

def train_S(args,train_loader,S_model,T_model,optimizer,criterion):
    S_model.train()
    T_model.eval()
    print("================================ Train epoch {} student model =====================================".format(args.epoch_S))
    for image_lst,image_dia_lst,label_lst,label_dia_lst,res_label_dia_lst,image_name_lst in train_loader:

        image = image_lst.float().to(args.device) 
        image_dia = image_dia_lst.float().to(args.device)
        label_dia = label_dia_lst.float().to(args.device)
        label = label_lst.float().to(args.device)
        res_label_dia = res_label_dia_lst.float().to(args.device)

        optimizer.zero_grad()

        with torch.no_grad():
            T_label = T_model(image_dia)
            Feas_teacher = T_model.feas 
            
        pred = S_model(image)
        Feas_student = S_model.feas

        pred_BCE = criterion["BCEloss"](pred, label) #about possibility,all inputs are the same shape!
        pred_dice = criterion["DiceLoss"](pred, label)
        KD_loss = Compute_KDloss(Feas_teacher,Feas_student,label)
        loss = 0.5 * pred_BCE + 0.5* pred_dice + args.KD_cof * KD_loss

        loss.backward()
        optimizer.step()



