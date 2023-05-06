import torch
import torchvision
from datasets_m import HwrOnPrintedDataset
from torch.utils.data import DataLoader
import os
import torch.nn.functional as F


def load_checkpoint(checkpoint, model):
    print('=> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])

def save_chekpoint(state, filename='my_checkpoint.pth.tar'):
    print('=> Saving checkpoint')
    torch.save(state, filename)

def get_loaders(
        train_image_path,
        train_image_info_filename,
        val_image_path,
        val_image_info_filename,
        train_transform,
        val_transform,
        batch_size,
        num_workers,
        pin_memory
):
    train_ds = HwrOnPrintedDataset(
        path=train_image_path, 
        info_filename=train_image_info_filename, 
        transform=train_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    val_ds = HwrOnPrintedDataset(
        path=val_image_path,
        info_filename=val_image_info_filename,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device='cuda'):
    num_correct = 0
    num_pixels = 0
    model.eval()
    # dice_score = 0

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device=device)
            targets = targets.to(device=device)
            predictions = torch.sigmoid(model(data))
            predictions = (predictions > 0.5).float()
            num_correct += (predictions == targets).sum()
            num_pixels += torch.numel(predictions)

            # for binary only
            # dice_score += (2 * (predictions * targets).sum()) / ((predictions + targets).sum() + 1e-8)
    
    print(f'Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:2f}')

    model.train()

def save_predictions_as_images(
    loader, 
    model, 
    folder='saved_images', 
    device='cuda'
):
    os.makedirs(folder, exist_ok=True)

    model.eval()
    for idx, (data, targets) in enumerate(loader):
        data = data.to(device=device)
        with torch.no_grad():
            predictions = torch.sigmoid(model(data))
            predictions = (predictions > 0.5).float()

        print(predictions.shape, targets.shape)

        def padding(x):
            p3d = (0, 0, 0, 0, 0, 1) # pad by (0, 0), (0, 0), and (0, 1)
            return F.pad(x, p3d, "constant", 0.0)
        
        predictions1 = padding(predictions)
        targets1 = padding(targets)

        print(predictions1.shape, targets1.shape)

        torchvision.utils.save_image(predictions1, os.path.join(folder, f'pred_{idx}.png'))
        torchvision.utils.save_image(targets1, os.path.join(folder, f'trgt_{idx}.png'))
