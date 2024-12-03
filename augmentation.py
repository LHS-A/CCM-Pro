# -- coding: utf-8 --
import cv2
import matplotlib.pyplot as plt
import random
import albumentations as A
import torchvision.transforms as transforms

aug = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12,12))

def flip_3(image_dia,image,label,flip_code = random.choice([0,1,-1]),p = random.random()):
    if p>0.5:
        flipped_image_dia =cv2.flip(image_dia, flip_code)
        flipped_image =cv2.flip(image, flip_code)
        flipped_label =cv2.flip(label, flip_code)

    else:
        flipped_image_dia = image_dia
        flipped_image = image
        flipped_label = label

    return flipped_image_dia,flipped_image,flipped_label


def apply_augmentations(image_dia,image,label):
    transform = A.Compose([
        A.ColorJitter(always_apply=False, p=0.7, brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
        A.GaussNoise(always_apply=False, p=0.7, var_limit=(10.0, 50.0), per_channel=True, mean=0.0),
        A.GaussianBlur(always_apply=False, p=0.7, blur_limit=(3, 7), sigma_limit=(0.0, 0)),
        A.HueSaturationValue(always_apply=False, p=0.7, hue_shift_limit=(-20, 20), sat_shift_limit=(-30, 30), val_shift_limit=(-20, 20)),
        A.RandomBrightnessContrast(always_apply=False, p=0.7, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True),
        # A.CoarseDropout(always_apply=False, p=0.7, max_holes=8, max_height=8, max_width=8, min_holes=8, min_height=8, min_width=8),
        # A.PixelDropout(always_apply=False, p=0.7, dropout_prob=0.01, per_channel=0)
    ])
    #概率执行翻转函数
    image_dia, image,label = flip_3(image_dia,image,label)
    augmented_image = transform(image=image)['image']

    return image_dia.squeeze(), augmented_image.squeeze(),label.squeeze()

