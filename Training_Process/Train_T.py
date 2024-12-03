# -- coding: utf-8 --
def train_T(args,train_loader,T_model,optimizer,criterion):
    T_model.train()
    print("================================ Train epoch {} teacher model =====================================".format(args.epoch_T))
    for image_lst,image_dia_lst,label_lst,label_dia_lst,res_label_dia_lst,image_name_lst in train_loader:
        image_dia = image_dia_lst.float().to(args.device)
        label = label_lst.float().to(args.device)

        optimizer.zero_grad()
        
        pred = T_model(image_dia)

        pred_BCE = criterion["BCEloss"](pred, label) 
        pred_dice = criterion["DiceLoss"](pred, label) 
        loss = 0.5 * pred_BCE + 0.5* pred_dice 

        loss.backward(retain_graph=False)
        optimizer.step()