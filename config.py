# -- coding: utf-8 --
"""
@Model:CCM-Pro   
@Time:2023/7/9
@Author:lihongshuo
"""
import os
from Dataset_List import DatasetParameters
import numpy as np
import torch
 
class Params(): 
    def __init__(self):
        self.root_path = r"/home/imed/personal/LHS/"
        self.model_name = "CCM-Pro"
        self.model_path = os.path.join(self.root_path, self.model_name) 
        self.dataset = "CORN_3_cell" ## DRIVE CCM2 CCM1 CORN_3_cell CORN_3 Stroma ACDC MoNuSeg 
        self.content = "_propose"  
        # Dataset params. 
        dataset_params = DatasetParameters(self.dataset)
        self.roi_size = [dataset_params.parameters["roi_size"][0],dataset_params.parameters["roi_size"][1]] # Crop size
        self.crop = dataset_params.parameters["crop"] # open crop 
        self.input_dim = dataset_params.parameters["input_dim"] # input channels
        self.num_classes = dataset_params.parameters["num_classes"] # output channels
        dataset_params = DatasetParameters(self.dataset)
        print(dataset_params.parameters)
        
        self.device_ids = [0] 
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.count_KD = 0 # Record the number of stage.
        self.value_thed = 60 # E_{max}^S for student in each training stage. 
        self.teacher_thed = 20 # E_{max}^T for teacher in each training stage. 
        self.KD_cof = 0.8 # hyper-parameters to trade off loss.
        self.checkpoint = False 
        self.dialated_pixels_list = [6,5,4,3,2,1,0.1]
        self.enhance_mode_S = None  # Use augmentations for teacher/student or not. S denotes student, T denotes teacher.
        self.enhance_mode_T = None 

        self.seed_random = 42
        self.best_dice_T = 0  
        self.best_dice_S = 0
        self.best_loss = np.inf 
        self.epoch_T = 0
        self.epoch_S = 0
        self.beta = 0
        self.epochs_T = len(self.dialated_pixels_list) * self.value_thed  + 88 # The total training epochs of teacher model.  
        self.epochs_S = len(self.dialated_pixels_list) * self.teacher_thed + 88 # The total training epochs of student model.
        self.best_epoch = 0 
        self.mark_T_epoch = 0 # The teacher training epochs at each stage!
        self.mark_S_epoch = 0 # The student training epochs at each stage!

        self.train_teacher = True 
        self.init_lr_S = 1e-4
        self.init_lr_T = 1e-4
        self.data_path = r"/home/imed/personal/LHS/CCM-Pro/Dataset/" + self.dataset

        self.train_batch = 4 
        self.val_batch = 4
        self.test_batch = 4

        self.S_Bestmodel_path = r"/home/imed/personal/LHS/CCM-Pro/model_weight/best_model/S/" + self.env_name_S 
        self.T_Bestmodel_path = r"/home/imed/personal/LHS/CCM-Pro/model_weight/best_model/T/" + self.env_name_T 
        os.makedirs(self.S_Bestmodel_path, exist_ok=True) 
        os.makedirs(self.T_Bestmodel_path, exist_ok=True)
    