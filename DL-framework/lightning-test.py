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
    from losses import BCEDiceLoss, TverskyLoss
    from models import Unet


    # MODEL
    model = Unet(
        features=[64, 128, 256],
        # loss=BCEDiceLoss(eps=1.0, activation=None))
        loss=TverskyLoss(alpha=0.5, beta=0.5)
    )
    model.configure_optimizers(lr=1e4)

    # DATA MODULE
    dm = HwrOnPrintedDataModule(
        path={
            'train': os.path.join(ds_path, 'train/result'),
            'val': os.path.join(ds_path, 'val/result'),
            'test': os.path.join(ds_path, 'test/result')
        },
        batch_size=4,
        num_workers=8,
        pin_memory=True if torch.cuda.is_available() else False)

    # UTILS
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=logs_path)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='loss/val',
        dirpath=checkpoints_path, #os.path.join(tb_logger.log_dir, 'checkpoints'), #'checkpoints',
        filename='unet-epoch{epoch:02d}-val_loss{loss/val:.2f}',
        auto_insert_metric_name=False,
        save_top_k=10
    )
    save_images_callback = SaveImagesCallback(
        image_dir=os.path.join(tb_logger.log_dir, 'images'),
        max_number=5,
        pad=(2, 0)
    )
    load_images_to_tb = LoadImageToTensorBoard(
        max_number=5,
        pad=(2, 0)
        )

    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    # TRAINER
    trainer = pl.Trainer(
        gpus=[3, 4],
        max_epochs=5, 
        log_every_n_steps=10, 
        callbacks=[checkpoint_callback, save_images_callback, load_images_to_tb, lr_logger], 
        logger=tb_logger)
    trainer.fit(model, dm)
