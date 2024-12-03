import os
import shutil
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM) 
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]

    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(out_shape[1]):
            posmask = img_gt[b].astype(bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                sdf[boundary==1] = 0
                sdf[sdf < 0] = 0 
                # print(np.unique(sdf))
                normalized_sdf[b][c] = sdf
                # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
                assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf

def get_SDF_data(image,label,beta):
    if len(image.shape) == 3:
        image = np.transpose(image, (2,0,1)) #[C,H,W]
        image = image[np.newaxis,:,:,:] # [1,C,H,W]
    else:
        image = image[np.newaxis,np.newaxis,:,:] # [1,1,H,W]
    label = label.squeeze()
    label = label[np.newaxis,np.newaxis,:,:] # [1,1,H,W]
    SDF_label = compute_sdf(label, label.shape)
    SDF_label = np.exp(-beta*SDF_label) 
    SDF_image = (image * SDF_label).astype(np.uint8).squeeze()
    SDF_label = SDF_label.squeeze() * 255
    SDF_label = SDF_label.astype(np.uint8)
    SDF_label = np.where(SDF_label > 128, 255, 0)
    res_SDF_label = 255 - SDF_label
    res_SDF_label[res_SDF_label > 0] = 255 
    if len(SDF_image.shape) == 3:
        SDF_image = np.transpose(SDF_image, (1,2,0)) #[C,H,W] -> [H,W,C] 

    return SDF_image,SDF_label,res_SDF_label

def sample_number(index):
    # seed = random.random()
    # if 0.5 < seed < 1:
    #     number = index
    # else:
    #     number = index - 1
    number = index

    return number
 
def step_SDF_dataloader(args,image,label):
    
    if args.count_KD == 0:
        args.beta = args.dialated_pixels_list[0]
        image,label,res_label = get_SDF_data(image,label,args.beta)
 
    elif args.count_KD == 1:
        index = sample_number(args.count_KD)
        args.beta = args.dialated_pixels_list[index]
        image,label,res_label = get_SDF_data(image,label,args.beta) 

    elif args.count_KD == 2:
        index = sample_number(args.count_KD)
        args.beta = args.dialated_pixels_list[index]
        image,label,res_label = get_SDF_data(image,label,args.beta) 

    elif args.count_KD == 3:
        index = sample_number(args.count_KD)
        args.beta = args.dialated_pixels_list[index]
        image,label,res_label = get_SDF_data(image,label,args.beta) 

    elif args.count_KD == 4:
        index = sample_number(args.count_KD)
        args.beta = args.dialated_pixels_list[index]
        image,label,res_label = get_SDF_data(image,label,args.beta) 

    elif args.count_KD == 5:
        index = sample_number(args.count_KD)
        args.beta = args.dialated_pixels_list[index]
        image,label,res_label = get_SDF_data(image,label,args.beta) 

    elif args.count_KD == 6:
        index = sample_number(args.count_KD)
        args.beta = args.dialated_pixels_list[index]
        image,label,res_label = get_SDF_data(image,label,args.beta) 

    elif args.count_KD == 7:
        index = sample_number(args.count_KD)
        args.beta = args.dialated_pixels_list[index]
        image,label,res_label = get_SDF_data(image,label,args.beta) 
    
    return image,label,res_label
