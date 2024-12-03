# -- coding: utf-8 --
import random 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import cv2
from augmentation import *
from utils import *
from FIND import *
from copy import deepcopy
transform_tensor = transforms.ToTensor()

def read_datasets(mode,args):
    images = []
    labels = []
    other_images = []

    if mode == "train":
        train_folder = os.path.join(args.data_path,'train')
        image_folder = os.path.join(train_folder, "image")
        images_name = os.listdir(image_folder)
        label_folder = os.path.join(train_folder, "label")
        
    elif mode == "val":
        val_folder = os.path.join(args.data_path,'val')
        image_folder = os.path.join(val_folder, "image")
        images_name = os.listdir(image_folder) 
        label_folder = os.path.join(val_folder, "label")         
            
    elif mode == "test":
        test_folder = os.path.join(args.data_path,'test')
        image_folder = os.path.join(test_folder, "image")
        images_name = os.listdir(image_folder) 
        label_folder = os.path.join(test_folder, "label")
    
    for name in images_name:
        img_path = os.path.join(image_folder, name)
        images.append(img_path)
        label_path = os.path.join(label_folder, name)
        labels.append(label_path)

    random_labels = deepcopy(labels)

    return images, labels, images_name, random_labels

class MyDataset(Dataset):
    def __init__(self,args,mode="train"):
        self.args = args
        self.mode = mode
        self.images, self.labels, self.images_name, self.random_labels = read_datasets(self.mode,args)
        random.shuffle(self.random_labels)

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.labels[index]
        image_name = self.images_name[index]
        random_label_path = self.random_labels[index]
    
        image = cv2.imread(image_path)
       
        label = cv2.imread(label_path,0)
        random_label = cv2.imread(random_label_path,0)

        label = label[:,:,np.newaxis]
        random_label = random_label[:,:,np.newaxis]
        
        if self.args.beta != None:
            image_dia,label_dia,res_label_dia = step_SDF_dataloader(self.args,image,label) 
            
        if random.random() < 0.5:
            beta = random.uniform((len(self.args.dialated_pixels_list)-self.args.count_KD),self.args.count_KD)
            image_dia,_,_ = get_SDF_data(image, cv2.add(label,random_label), beta) 

        if len(np.unique(label)) == 1: 
            label_dia = np.ones_like(label).squeeze() #[H,W]
            res_label_dia = np.ones_like(label).squeeze() #[H,W]     
    
        if self.mode == self.args.enhance_mode_S and self.args.train_teacher == False:
            image_dia,image,label = apply_augmentations(image_dia,image,label) 
        
        image = transform_tensor(image) 
        label = transform_tensor(label)
        image_dia = transform_tensor(image_dia)
        label_dia = transform_tensor(label_dia)
        label_dia = torch.where(label_dia > 0,torch.tensor(1),torch.tensor(0))
        res_label_dia = transform_tensor(res_label_dia) 
        res_label_dia = torch.where(res_label_dia > 0,torch.tensor(1),torch.tensor(0))
        
        return image,image_dia,label,label_dia,res_label_dia,image_name

    def __len__(self):
        assert len(self.images) == len(self.labels) 
        return len(self.images)


class Data_loader():
    def __init__(self):
        pass

    def load_train_data(self, args,batch_size):
        dataset = MyDataset(args,mode="train")
        train_loader = DataLoader(dataset, batch_size, shuffle=True, pin_memory=False)
        return train_loader
    
    def load_val_data(self,args,batch_size):
        dataset = MyDataset(args,mode="val")
        val_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True, pin_memory=False)
        return val_loader
    
    def load_test_data(self,args,batch_size):
        dataset = MyDataset(args,mode="test")
        test_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True, pin_memory=False)
        return test_loader
    

if __name__ == "__main__":
    root_path = r"G:\VSCODE\fundus_vessels_seg\CORN3"
    dataloader = Data_loader(root_path)
    train_loader = dataloader.load_train_data(1)
    test_loader = dataloader.load_test_data(1)
    print(len(train_loader))
    print(len(test_loader))




        
