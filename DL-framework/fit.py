class HP:
    def __init__(self, model_type, loss, loss_params, batch_size, image_height, image_width, ds_part, epoches, features) -> None:
        self.model_type = model_type
        self.loss = loss(**loss_params)
        self.loss_params = loss_params
        self.loss_name = self.loss.__name__
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.ds_part = ds_part
        self.epoches = epoches
        self.features = features
        self.model_name = f'{model_type}-{features[-1]}_{features[0]}-{image_height}_{image_width}-{self.loss_name}'
    
    def dict_for_log(self):
        return {
            'MODEL_NAME': self.model_name,
            'LOSS': self.loss_name, 
            'LOSS_PARAMS': str(self.loss_params), 
            'EPOCHES': self.epoches,
            'FEATURES': str(self.features),
            'DS_SIZE': self.ds_part,
            'IMAGE_HEIGHT': self.image_height,
            'IMAGE_WIDTH': self.image_width
        }
    
    def __str__(self):
        return str(self.dict_for_log())


def fit(hparams: HP = None):
    print('Starting FIT with hyperparams:')
    print(str(hparams))

    # MODEL
    model = Unet(
        features=hparams.features,
        loss=hparams.loss
    )

    # DATA MODULE
    dm = HwrOnPrintedDataModule(
        path={
            'train': os.path.join(ds_path, 'train/result'),
            'val': os.path.join(ds_path, 'val/result'),
            'test': os.path.join(ds_path, 'test/result')
        },
        batch_size=hparams.batch_size,
        num_workers=10,
        pin_memory=True if torch.cuda.is_available() else False,
        image_height=hparams.image_height,
        image_width=hparams.image_width,
        ds_part=hparams.ds_part
        )


    # UTILS
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=logs_path)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='_loss/val',
        dirpath=os.path.join(tb_logger.log_dir, hparams.model_name),
        filename='epoch{epoch:02d}-val_loss{_loss/val:.2f}',
        auto_insert_metric_name=False,
        save_top_k=3
    )
    save_images_callback = SaveImagesCallback(
        image_dir=os.path.join(tb_logger.log_dir, 'images'),
        max_number=2,
        pad=(2, 0)
    )
    load_images_to_tb = LoadImageToTensorBoard(
        max_number=10,
        pad=(2, 0)
        )

    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    early_stopping = pl.callbacks.EarlyStopping('_loss/val', mode='min', min_delta=0.02, patience=3)

    # TRAINER
    trainer = pl.Trainer(
        gpus=[3],
        max_epochs=hparams.epoches, 
        log_every_n_steps=2, 
        callbacks=[checkpoint_callback, load_images_to_tb, lr_logger, early_stopping], 
        # callbacks=[checkpoint_callback, save_images_callback, load_images_to_tb, lr_logger, early_stopping], 
        logger=tb_logger)
    
    # dm.setup('fit')
    trainer.fit(model, dm)
    trainer.test(model, dm)
    
    tb_logger.log_hyperparams(hparams.dict_for_log())
    tb_logger.finalize("success")


if __name__ == '__main__':

    import os
    cuda_devices = '0, 1, 2, 3, 4, 5, 6, 7'
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices

    ds_path = '/home/kudrjavtseviv/data/dataset_1'
    # ds_path = '/Users/vankudr/Documents/НИР-data/dataset_1'
    model_data_path = '/home/kudrjavtseviv/DL-framework/.data'

    checkpoints_path = os.path.join(model_data_path, 'checkpoints')
    logs_path = os.path.join(model_data_path, 'logs')

    import torch

    import gc
    torch.cuda.empty_cache()
    gc.collect()

    import pytorch_lightning as pl

    from callbacks import SaveImagesCallback, LoadImageToTensorBoard

    from datasets_m import HwrOnPrintedDataModule
    from losses import DiceLoss, BCEDiceLoss, TverskyLoss, FocalTverskyLoss
    from models import Unet, ResNet
    import torch.nn as nn

    DS_PART = 0.01
    IMAGE_HEIGHT = 800
    IMAGE_WIDTH = 600
    BATCH_SIZE = 2
    EPOCHES = 5

    configs = [
        HP(
            model_type='Unet',
            loss=DiceLoss,
            loss_params={'activation': None},
            batch_size=BATCH_SIZE,
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
            ds_part=DS_PART,
            epoches=EPOCHES,
            features=[64, 128, 256, 512]
        ),
        HP(
            model_type='Unet',
            loss=DiceLoss,
            loss_params={'activation': None},
            batch_size=BATCH_SIZE,
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
            ds_part=DS_PART,
            epoches=EPOCHES,
            features=[32, 64, 128, 256]
        ),
        HP(
            model_type='Unet',
            loss=DiceLoss,
            loss_params={'activation': None},
            batch_size=BATCH_SIZE,
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
            ds_part=DS_PART,
            epoches=EPOCHES,
            features=[16, 32, 64, 128]
        ),
        HP(
            model_type='Unet',
            loss=FocalTverskyLoss,
            loss_params={'alpha': 0.8, 'beta': 0.2, 'gamma': 2},
            batch_size=BATCH_SIZE,
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
            ds_part=DS_PART,
            epoches=EPOCHES,
            features=[64, 128, 256, 512]
        ),
        HP(
            model_type='Unet',
            loss=FocalTverskyLoss,
            loss_params={'alpha': 0.8, 'beta': 0.2, 'gamma': 2},
            batch_size=BATCH_SIZE,
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
            ds_part=DS_PART,
            epoches=EPOCHES,
            features=[32, 64, 128, 256]
        ),
        HP(
            model_type='Unet',
            loss=FocalTverskyLoss,
            loss_params={'alpha': 0.8, 'beta': 0.2, 'gamma': 2},
            batch_size=BATCH_SIZE,
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
            ds_part=DS_PART,
            epoches=EPOCHES,
            features=[16, 32, 64, 128]
        ),
    ]

    for config in configs[:1]:
        fit(config)
