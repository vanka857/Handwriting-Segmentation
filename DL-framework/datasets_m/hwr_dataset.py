from .base_dataset import BaseDataset
import pandas as pd
import cv2
import os
import numpy as np
import torch


def load_image(file_path):
    return cv2.imread(file_path)


class HwrOnPrintedDataset(BaseDataset):
    def __init__(self, path, info_filename, info_image_column='image', info_mask_column='label', transform=None):
        self.path = path
        self.info_filename = info_filename
        self.info_image_column = info_image_column
        self.info_mask_column = info_mask_column
        self.transform = transform

        print(transform)
        
        self.meta = pd.read_csv(info_filename, sep='\t')

    def __len__(self):
        return 100
        return len(self.meta)

    def __getitem__(self, index):
        item = self.meta.iloc[index]
        image = load_image(os.path.join(self.path, item[self.info_image_column]))[:, :, 0:1] # take only 1st channel
        mask = load_image(os.path.join(self.path, item[self.info_mask_column]))[:, :, 1:3] # take 2nd and 3rd dim 

        mask[mask == 255] = 1
        mask = mask.astype(np.float32)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
        
        mask1 = torch.permute(mask, (2, 0, 1)) # fix albumentations bug with channels roll

        return image, mask1
