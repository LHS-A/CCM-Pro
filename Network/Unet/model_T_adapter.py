# -- coding: utf-8 --
import torch.nn as nn
from torch.nn import functional as F
import torch
import torch.optim as optim

class DW_Conv(nn.Module):
    """
    DWConv + PointWiseConv
    """
    def __init__(self, in_planes, out_planes, stride=1):
        super(DW_Conv, self).__init__()
        self.conv_DW1 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride,padding=1, groups=in_planes, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU()
        )
        self.conv_PW1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,padding=0,bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )
    
    def forward(self, x):
        out = self.conv_DW1(x)
        out = self.conv_PW1(out)
        return out

class Structure_Adapter(nn.Module):
    def __init__(self, in_channels, mid_channels = 64):
        super().__init__() 
        self.Down = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.activation = DW_Conv(mid_channels,mid_channels)
        self.Up = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        self.scale_adapter = 1

    def forward(self, x):
        return (self.Up(self.activation(self.Down(x)))) * self.scale_adapter

class Net_T(nn.Module):
    def __init__(self, n_channels, n_classes,mode_code = "stroma",bilinear=True):
        super(Net_T, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.feas = []
        self.mode_code = mode_code
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

        self.S_adapter1 = Structure_Adapter(64)
        self.S_adapter2 = Structure_Adapter(128)
        self.S_adapter3 = Structure_Adapter(256)
        self.S_adapter4 = Structure_Adapter(512)
        self.S_adapter5 = Structure_Adapter(1024)
        self.S_adapter6 = Structure_Adapter(512)
        self.S_adapter7 = Structure_Adapter(256)
        self.S_adapter8 = Structure_Adapter(128)
        self.S_adapter9 = Structure_Adapter(64)

        if self.mode_code == "nerve":
            self.nerve_out = OutConv(64,n_classes)
        else:
            self.stroma_out = OutConv(64,n_classes)
        
    def forward(self, image):
        self.feas.clear()
        image1 = self.inc(image)
        adapter1 = self.S_adapter1(image1)
        image1= image1 + adapter1
        self.feas.append(adapter1) 
        
        image2 = self.down1(image1)
        adapter2 = self.S_adapter2(image2)
        image2= image2 + adapter2
        self.feas.append(adapter2)

        image3 = self.down2(image2)
        adapter3 = self.S_adapter3(image3)
        image3 = image3 + adapter3
        self.feas.append(adapter3)
        
        
        image4 = self.down3(image3)
        adapter4 = self.S_adapter4(image4)
        image4 = adapter4 + image4
        self.feas.append(adapter4)
        
        image5 = self.down4(image4)
        adapter5 = self.S_adapter5(image5)
        image5 = adapter5 + image5
        self.feas.append(adapter5)
        
        decoder11 = self.up11(image5, image4)
        adapter6 = self.S_adapter6(decoder11)
        decoder11 = decoder11 + adapter6
        self.feas.append(adapter6)
        
        decoder12 = self.up12(decoder11, image3)
        adapter7 = self.S_adapter7(decoder12)
        decoder12 = decoder12 + adapter7
        self.feas.append(adapter7)
        
        decoder13 = self.up13(decoder12, image2)
        adapter8 = self.S_adapter8(decoder13)
        decoder13 = decoder13 + adapter8
        self.feas.append(adapter8)
        
        decoder14 = self.up14(decoder13, image1)
        adapter9 = self.S_adapter9(decoder14)
        decoder14 = decoder14 + adapter9
        self.feas.append(adapter9)
        
        if self.mode_code == "nerve":
            pred = self.nerve_out(decoder14)
        else:
            pred = self.stroma_out(decoder14)
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
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,padding=1, bias=False)
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
    
