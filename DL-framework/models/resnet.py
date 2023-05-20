import pytorch_lightning as pl
import torch.optim as optim
from torchvision.models import resnet50

# Old weights with accuracy 76.130%
# resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

class ResNet(pl.LightningModule):
    def __init__(self, loss=None) -> None:
        super(ResNet, self).__init__()
        self.loss_fn = loss
        self.model = resnet50(pretrained=True)

    def forward(self, x):
        return self.model(x)
    
    def predict_step(self, batch, batch_idx):
        data, targets = batch
        pred = self.forward(data)
        return pred

    def sharing_step(self, batch, batch_idx):
        data, targets = batch
        predictions = self.forward(data).softmax()

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
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=1, cooldown=0)
        return [optimizer]#, [{"scheduler": scheduler, "interval": "epoch", "monitor": "loss/val"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)