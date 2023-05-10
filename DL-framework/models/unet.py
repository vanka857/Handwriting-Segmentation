from .base_model import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
import pytorch_lightning as pl


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

        # bottleneck layer
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # up part
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))
        
        # final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # LOSS
        self.loss_fn = nn.BCELoss(reduction='mean') if loss is None else loss
    
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
        pred = self(data)
        return pred

    def sharing_step(self, batch, batch_idx):
        data, targets = batch
        predictions = self(data).sigmoid()

        targets = targets[:, 2:3]
        # print(predictions.shape, targets.shape)
        # print(predictions, targets.max)

        loss = self.loss_fn(predictions, targets)
        return predictions, loss

    def training_step(self, batch, batch_idx):
        _, loss = self.sharing_step(batch, batch_idx)

        # log
        self.log('loss/train', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        predictions, val_loss = self.sharing_step(batch, batch_idx)

        # log
        self.log('loss/val', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return predictions
    
    def configure_optimizers(self, lr=1e-4):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        # optimizer = optim.RAdam(self.parameters(), lr = 0.005)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=1, cooldown=0)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": "loss/val"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)
