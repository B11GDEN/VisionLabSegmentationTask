import argparse
import nip

import segmentation
from lightning import Trainer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from segmentation.callbacks import ModelSaver


def parse_args():
    parser = argparse.ArgumentParser(description="Training Cycle")
    parser.add_argument("--config", type=str, help="# path to config")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    segmentation.__init__(__name__)

    # parse config
    args = parse_args()
    config = nip.load(args.config)

    # datamodule
    datamodule = config['datamodule']

    # lit_module
    lit_module = config['lit_module']

    # Callbacks
    checkpoint_callback = ModelSaver(
        monitor='val_average_dice',
        filename='{epoch}-{val_average_dice:.3f}',
        save_top_k=5,
        mode='max',
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    rich_progress = RichProgressBar()

    # Logger
    logger = WandbLogger(
        project=config['project'],
        name=config['name'],
    )

    # Trainer
    trainer = Trainer(
        check_val_every_n_epoch=config['check_val_every_n_epoch'],
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        max_epochs=config['max_epochs'],
        accumulate_grad_batches=config['accumulate_grad_batches'],
        callbacks=[rich_progress, lr_monitor, checkpoint_callback],
        logger=logger,
        accelerator=config["accelerator"],
        devices=config["devices"]
    )

    # Find best learning rate
    # tuner = Tuner(trainer)
    # tuner.lr_find(lit_module, datamodule,
    #               max_lr=1e-2, min_lr=1e-8, num_training=1000)

    # Train Model
    trainer.fit(lit_module, datamodule)

    x = 0
