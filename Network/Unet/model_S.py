# -- coding: utf-8 --
import torch.nn as nn
from torch.nn import functional as F
import torch

class Net_S(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, sparsity=0.3):
        super(Net_S, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.feas = []
        #正常Unet编码器分支
        self.inc = Double_Conv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512,1024)
        
        self.up11 = Up(1024, 512)
        self.up12 = Up(512, 256)
        self.up13 = Up(256, 128)
        self.up14 = Up(128, 64)
    
        self.nerve_out = OutConv(64, n_classes)
        

    def forward(self, image):
        self.feas.clear()
        image1 = self.inc(image)
        self.feas.append(image1)

        image2 = self.down1(image1)
        self.feas.append(image2)

        image3 = self.down2(image2)
        self.feas.append(image3)

        image4 = self.down3(image3)
        self.feas.append(image4)

        image5 = self.down4(image4)
        self.feas.append(image5)

        decoder11 = self.up11(image5, image4)
        self.feas.append(decoder11)
    
        decoder12 = self.up12(decoder11, image3)
        self.feas.append(decoder12)

        decoder13 = self.up13(decoder12, image2)
        self.feas.append(decoder13)

        decoder14 = self.up14(decoder13, image1)
        self.feas.append(decoder14)

        pred = self.nerve_out(decoder14)
        self.feas.append(pred)

        return pred
    
class Double_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn = True, sparsity=0.3):
        super().__init__()
        self.use_bn = use_bn
        # 分散开写是因为要区分BN与conv，我们冻结conv，不改变BN的值，后期可作为消融实验
        #sequence就是串联执行，如果并联执行必须要分开写两次，自定义卷积层可指定权重参数（掩膜冻结卷积层部分参数）
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
        else:#bn参数是否会更新，可直接控制
            self.bn1 = nn.BatchNorm2d(out_channels, track_running_stats=False, affine=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,bias=False)
        if self.use_bn:
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.bn2 = nn.BatchNorm2d(out_channels, track_running_stats=False, affine=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        #针对模块堆叠网络，传入模块名称，然后内部依次读取卷积参数即可；bn可以选择使用或者不使用
        #down64.conv1.weight_mask   down64.conv2.weight_mask 
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        return x

class Down(nn.Module):                                                                                                                                          
    def __init__(self, in_channels, out_channels, use_bn = True, sparsity=0.3):
        super().__init__()
        # 分散开写是因为要区分BN与conv，我们冻结conv，不改变BN的值，后期可作为消融实验
        self.maxpool = nn.MaxPool2d(2)
        self.double_conv = Double_Conv(in_channels, out_channels, use_bn = True, sparsity=0.5)

    def forward(self, x):
        #针对模块堆叠网络，传入模块名称，然后内部依次读取卷积参数即可；bn可以选择使用或者不使用
        #down64.conv1.weight_mask   down64.conv2.weight_mask 
        x = self.maxpool(x)
        x = self.double_conv(x) 
        
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels,use_bn = True, sparsity=0.3):
        super().__init__()
        self.use_bn = use_bn
        #上采样不要使用反卷积，会产生棋盘格效应
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
        else:
            self.bn1 = nn.BatchNorm2d(out_channels, track_running_stats=False, affine=False)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if self.use_bn:
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.bn2 = nn.BatchNorm2d(out_channels, track_running_stats=False, affine=False)
        self.relu = nn.ReLU(inplace=True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #不要以,结尾，不然会被报错为元组对象无法执行
        self.upconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0,  bias=False)

    def forward(self,decoder, skip):
        #up64.conv1.weight_mask   up64.conv2.weight_mask   up64.upconv.weight_mask 偏置全部设置为false
        decoder = self.upconv(self.upsample(decoder))
        x = torch.cat([skip, decoder], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels,sparsity=0.3):
        super(OutConv, self).__init__()
        # 低层特征发生大量迁移现象，越往高层迁移越不明显，最后一层融合信息很关键，前期学的好后期融合差也不行，所以先约束一下，可消融实验
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0,  bias=False)

    def forward(self, x):
        #out.conv.weight_mask
        x = self.conv(x)
        return x
    

def get_interpolate_items(teacher_feas,label_dia,res_label_dia):
    label_dia = F.interpolate(label_dia,size = teacher_feas.shape[-2:],mode = "bilinear",align_corners=True)
    res_label_dia = F.interpolate(res_label_dia,size = teacher_feas.shape[-2:],mode = "bilinear",align_corners=True)
    
    return label_dia,res_label_dia

def Simloss(Fea_s,Fea_t,label,dia_label,KD_nothing):
# =====================================================================================================================
    Fea_s = Fea_s.pow(2).mean(1)
    Fea_t = Fea_t.pow(2).mean(1)
    mseloss = nn.MSELoss()
     # Get fore target field.
    Fea_t_fore = Fea_t * label
    Fea_s_fore = Fea_s * label
    # Get back target field.
    Fea_t_back = Fea_t * dia_label
    Fea_s_back = Fea_s * dia_label

# # =============================================== Matrix loss ==========================================================
    # The second way: compute the simplest way in mseloss! much more than cosL2!

    Fea_t_fore = (Fea_t_fore - Fea_t_fore.min()) / (Fea_t_fore.max() - Fea_t_fore.min())
    Fea_s_fore = (Fea_s_fore - Fea_s_fore.min()) / (Fea_s_fore.max() - Fea_s_fore.min())

    Fea_t_back = (Fea_t_back - Fea_t_back.min()) / (Fea_t_back.max() - Fea_t_back.min())
    Fea_s_back = (Fea_s_back - Fea_s_back.min()) / (Fea_s_back.max() - Fea_s_back.min())
    
    # New building tensor need to device first! features do not need! Nerve do something inplace,such as A[mask] = B[mask] !
    # mask_fore = torch.where(Fea_s_fore > Fea_t_fore,0,1).to(device) # if fore,then the more,the better
    # mask_back = torch.where(Fea_s_back < Fea_t_back,0,1).to(device) # if back,then the less,the better
    # # print("mask_back:",torch.unique(mask_fore)) 
   
    # Fea_s_fore = mask_fore * Fea_s_fore
    # Fea_t_fore = mask_fore * Fea_t_fore
    
    # Fea_t_back = mask_back * Fea_t_back
    # Fea_s_back = mask_back * Fea_s_back
    if KD_nothing == False:
        Simloss_fore = mseloss(Fea_s_fore,Fea_t_fore)
        # Simloss_back = mseloss(Fea_s_back,Fea_t_back)
        # simloss = Simloss_fore + Simloss_back 
        simloss = Simloss_fore
    else:
        Fea_s = (Fea_s - Fea_s.min()) / (Fea_s.max() - Fea_s.min())
        Fea_t = (Fea_t - Fea_t.min()) / (Fea_t.max() - Fea_t.min())
        simloss = mseloss(Fea_s,Fea_t)

    return simloss