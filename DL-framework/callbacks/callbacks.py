import torch
import os
import pytorch_lightning as pl
import torchvision
import torch.nn.functional as F


def get_images(preds, targets, pad):
    if pad is None:
        pad = (0, 1)

    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    def padding(x):
        p3d = (0, 0, 0, 0) + pad # pad by (0, 0), (0, 0), and (0, 1)
        return F.pad(x, p3d, "constant", 0.0)
    
    preds = padding(preds)
    # targets = padding(targets)

    return preds, targets


class SaveImagesCallback(pl.Callback):
    def __init__(self, image_dir, max_number=10, pad=None):
        self.max_number = max_number
        self.image_dir = image_dir
        self.pad = pad
        os.makedirs(image_dir, exist_ok=True)
        self.state = {
            'image_dir': image_dir,
            'max_number': max_number,
            'current_number': 0
        }
        
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx < self.max_number:
            self.state['current_number'] = batch_idx
            _, targets = batch
            preds, targets = get_images(outputs, targets, pad=self.pad)

            self.epoch = trainer.current_epoch

            torchvision.utils.save_image(preds, os.path.join(self.image_dir, f'batch_{batch_idx}_pred_epoch_{self.epoch}.png'))
            if self.epoch == 0:
                torchvision.utils.save_image(targets, os.path.join(self.image_dir, f'batch_{batch_idx}_trgt.png'))

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

    def state_dict(self):
        return self.state.copy()


class LoadImageToTensorBoard(pl.Callback):
    def __init__(self, max_number=10, pad=None):
        self.max_number = max_number
        self.pad = pad
        self.state = {
            'max_number': max_number,
            'current_number': 0
        }

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx < self.max_number:
            self.state['current_number'] = batch_idx
            _, targets = batch
            preds, targets = get_images(outputs, targets, pad=self.pad)
            preds_grid = torchvision.utils.make_grid(preds)

            self.epoch = trainer.current_epoch
            trainer.logger.experiment.add_image(f'batch_{batch_idx}/pred/epoch_{self.epoch}', preds_grid, 0) 

            if self.epoch == 0:
                targets_grid = torchvision.utils.make_grid(targets)
                trainer.logger.experiment.add_image(f'batch_{batch_idx}/trgt', targets_grid, 0)

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

    def state_dict(self):
        return self.state.copy()
