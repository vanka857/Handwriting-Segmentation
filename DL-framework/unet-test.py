import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
import torch
import os

from models import Unet
from utils import (
    load_checkpoint, 
    save_chekpoint, 
    get_loaders, 
    check_accuracy, 
    save_predictions_as_images, 
    train_fn
)


# Hyperparameters
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16
num_epochs = 3
num_workers = 0
image_height = 60
image_width = 42
pin_memory = True if torch.cuda.is_available() else False
load_model = False
train_image_path = '/Users/vankudr/Documents/НИР-data/dataset_1/train/result'
train_image_info_filename = os.path.join(train_image_path, '_info.csv')
val_image_path = '/Users/vankudr/Documents/НИР-data/dataset_1/val/result'
val_image_info_filename = os.path.join(val_image_path, '_info.csv')


def test():
    train_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
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

    val_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=1.0
            ),
            ToTensorV2()
        ]
    )

    model = Unet(in_channels=1, out_channels=2).to(device)
    loss_fn = nn.CrossEntropyLoss() # because 2 channels on output
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, val_loader = get_loaders(
        train_image_path,
        train_image_info_filename,
        val_image_path,
        val_image_info_filename,
        train_transform,
        val_transform,
        batch_size,
        num_workers,
        pin_memory
    )

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, device=device)

        # save model
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_chekpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=device)

        # save examples
        save_predictions_as_images(val_loader, model, folder='saved_images', device=device)
    

test()