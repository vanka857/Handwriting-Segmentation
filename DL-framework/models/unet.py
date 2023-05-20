from .base_model import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class Unet(pl.LightningModule):
    def __init__(self, in_channels=1, out_channels=1, features=[8, 16, 32], loss=None) -> None:
        super(Unet, self).__init__()

        # NET PARAMS
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # down part
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        print(1)

        # bottleneck layer
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        print(2)

        # up part
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        print(3)
        
        # final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        print(4)

        # LOSS
        self.loss_fn = loss

        # METRICS
        self.acc_fn = BinaryAccuracy()
        self.prec_fn = BinaryPrecision()
        self.rec_fn = BinaryRecall()

        # saving hyperparameters
        self.save_hyperparameters(ignore=['loss'], logger=False)
    
    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            # UP by ConvTranspose and Double conv on each iteration
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

    def predict_step(self, batch, batch_idx):
        data, targets = batch
        return self(data).sigmoid()

    def _sharing_step(self, batch, batch_idx):
        data, targets = batch
        predictions = self(data).sigmoid()

        targets = targets[:, 2:3]

        loss = self.loss_fn(predictions, targets)
        return [predictions, targets, loss]
    
    def test_step(self, batch, batch_idx):
        predictions, targets, test_loss = self._sharing_step(batch, batch_idx)

        test_acc = self.acc_fn(predictions, targets)
        test_prec = self.prec_fn(predictions, targets)
        test_rec = self.rec_fn(predictions, targets)

        # log
        self.log('_loss/test', test_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('_acc/test', test_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('_precision/test', test_prec, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('_recall/test', test_rec, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return predictions

    def training_step(self, batch, batch_idx):
        _, _, loss = self._sharing_step(batch, batch_idx)

        # log
        self.log('_loss/train', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        predictions, targets, val_loss = self._sharing_step(batch, batch_idx)

        val_acc = self.acc_fn(predictions, targets)
        val_prec = self.prec_fn(predictions, targets)
        val_rec = self.rec_fn(predictions, targets)

        # log
        self.log('_loss/val', val_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('_acc/val', val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('_precision/val', val_prec, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('_recall/val', val_rec, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return predictions
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1, cooldown=1, min_lr=1e-4)
        # return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": "_loss/val"}]
        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "_loss/val",
                },
            },
        )
