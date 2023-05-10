from .base_dataset import BaseDataset
import pandas as pd
import cv2
import os
import numpy as np
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader


def load_image(file_path):
    return cv2.imread(file_path)


class HwrOnPrintedDataset(BaseDataset):
    def __init__(self, path, info_filename, info_image_column='image', info_mask_column='label', transform=None):
        self.path = path
        self.info_filename = info_filename
        self.info_image_column = info_image_column
        self.info_mask_column = info_mask_column
        self.transform = transform
        
        self.meta = pd.read_csv(info_filename, sep='\t')

    def __len__(self):
        return 500
        return len(self.meta)

    def __getitem__(self, index):
        item = self.meta.iloc[index]
        image = load_image(os.path.join(self.path, item[self.info_image_column]))[:, :, 0:1] # take only 1st channel
        mask = load_image(os.path.join(self.path, item[self.info_mask_column])) # [:, :, 1:3] # take 2nd and 3rd dim 

        mask[mask == 255] = 1
        mask = mask.astype(np.float32)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
        
        mask1 = mask.permute(2, 0, 1) # fix albumentations bug with channels roll

        return image, mask1


class HwrOnPrintedDataModule(pl.LightningDataModule):
    def __init__(self, path, batch_size=32, num_workers=0, pin_memory=False):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.image_height = 640
        self.image_width = 420

        self.train_transform = A.Compose(
            [
                A.Resize(height=self.image_height, width=self.image_width),
                A.Rotate(limit=35, p=1.0),
                A.HorizontalFlip(p=0.2),
                A.VerticalFlip(p=0.2),
                A.Normalize(
                    mean=[0.0],
                    std=[1.0],
                    max_pixel_value=255.0
                ),
                ToTensorV2()
            ]
        )

        self.val_transform = A.Compose(
            [
                A.Resize(height=self.image_height, width=self.image_width),
                A.Normalize(
                    mean=[0.0],
                    std=[1.0],
                    max_pixel_value=255.0
                ),
                ToTensorV2()
            ]
        )

    def setup(self, stage: str):
        self.ds_train = HwrOnPrintedDataset(
            path=self.path['train'],
            info_filename=os.path.join(self.path['train'], '_info.csv'), 
            transform=self.train_transform
        )
        self.ds_val = HwrOnPrintedDataset(
            path=self.path['val'],
            info_filename=os.path.join(self.path['val'], '_info.csv'), 
            transform=self.val_transform
        )
        self.ds_test = HwrOnPrintedDataset(
            path=self.path['test'],
            info_filename=os.path.join(self.path['test'], '_info.csv'), 
            transform=self.val_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.ds_train, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True)

    def val_dataloader(self):
        return DataLoader(
            self.ds_val, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False)

    def test_dataloader(self):
        return DataLoader(
            self.ds_test, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False)

    # def predict_dataloader(self):
    #     return DataLoader(self.ds_test, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass
