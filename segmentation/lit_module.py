import math
import wandb
import matplotlib.pyplot as plt

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from lightning import LightningModule
from torchmetrics import Metric, MeanMetric, ConfusionMatrix
from torchmetrics.functional import dice as dice_score

from segmentation.data.transforms import LABEL2ORGAN
from segmentation.metrics import plot_confusion_matrix


class LitSegmenter(LightningModule):
    """
    `LightningModule` for segmentation task.

    Implements training and validation logic.

    Attributes:
        net (torch.nn.Module): Segmentation network for training, e.g. Unet, DeepLab, ...
        loss (torch.nn.Module): Segmentation loss, e.g. Dice, IoU, CrossEntropy, ...
        learning_rate (float): Initial learning rate, this attribute is also needed because of
            Lightning Tuner to tune the value
        val_dices (list[Metric]): array of dices coefficients, each dice refers to organ in `LABEL2ORGAN` dictionary
        confmat (Metric): class to compute confusion matrix
        matrix (torch.Tensor):  array to store confusion matrix value

    Methods:
        __init__: Initialize the `LitSegmenter` object
        forward: Forward pass
        training_step: Single training step on a batch of data
        on_validation_epoch_start: Lightning hook that is called when validation epoch starts
        validation_step: Single validation step on a batch of data
        on_validation_epoch_end: Lightning hook that is called when validation epoch ends
        configure_optimizers: Сonfigure optimizer nad sheduler

    """
    def __init__(
        self,
        net: torch.nn.Module,
        loss: torch.nn.Module,
        learning_rate: float = 3e-4,
    ) -> None:
        """
        Initialize a `LitSegmenter`.

        Args:
            net (torch.nn.Module): Segmentation network for training, e.g. Unet, DeepLab, ...
            loss (torch.nn.Module): Segmentation loss, e.g. Dice, IoU, CrossEntropy, ...
            learning_rate (float): Initial learning rate, Defaults to `3e-4`
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['net', 'loss', 'learning_rate'])

        self.net: torch.nn.Module = net
        self.loss: torch.nn.Module = loss

        # Need this variable for learning rate tuner
        self.learning_rate: float = learning_rate

        # Metrics
        self.val_dices: list[Metric] = [MeanMetric() for _ in LABEL2ORGAN.values()]

        # Variables for confusion Matrix
        self.confmat: Metric = ConfusionMatrix(task="multiclass", num_classes=len(LABEL2ORGAN)+1)
        self.matrix: torch.Tensor = torch.zeros((len(LABEL2ORGAN)+1, len(LABEL2ORGAN)+1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x (torch.Tensor): input tensor
        Returns:
            output (torch.Tensor): output tensor
        """
        return self.net(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Single training step. Compute prediction, loss and log them.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): Batch output from dataloader.
            batch_idx (int): number of batch

        Returns:
            loss (torch.Tensor): calculated loss
        """
        x, y = batch

        pred = self.forward(x)
        loss = self.loss(pred, y)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def on_validation_epoch_start(self) -> None:
        """
        Lightning hook that is called when validation epoch starts.

        Reset all dices metrics and confusion matrix
        """
        for metric in self.val_dices:
            metric.reset()
            metric.to(self.device)
        self.matrix = torch.zeros((len(LABEL2ORGAN)+1, len(LABEL2ORGAN)+1), device=self.device)

    def validation_step(self, batch, batch_idx) -> None:
        """
        Single validation step. Compute prediction, loss and log them. Then update dices and confusion matrix

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): Batch output from dataloader.
            batch_idx (int): number of batch
        """
        x, y = batch

        pred = self.forward(x)
        loss = self.loss(pred, y)
        self.log("val_loss", loss, prog_bar=True)

        # Compute Dice
        dices = dice_score(pred, y, average=None, num_classes=len(LABEL2ORGAN)+1)
        for dice, metric in zip(dices[1:], self.val_dices):
            dice = 1.0 if math.isnan(dice) else dice  # class doesn't occur in the preds or target
            metric.update(dice)

        # Compute Confusion Matrix
        self.matrix += self.confmat(pred, y)

    def on_validation_epoch_end(self) -> None:
        """
        Lightning hook that is called when validation epoch ends.

        Combine results from all batches, calculate dices and plot confusion matrix
        """
        # Log dice coefficients
        avr_dice = 0
        for organ, metric in zip(LABEL2ORGAN.values(), self.val_dices):
            dice = metric.compute()
            self.log(f"val_{organ}_dice", dice)
            avr_dice += dice
        avr_dice /= len(LABEL2ORGAN)
        self.log(f"val_average_dice", avr_dice, prog_bar=True)

        # Log Confusion Matrix
        fig, _ = plot_confusion_matrix(
            self.matrix.cpu().numpy(),
            figsize=(10, 8),
            cmap=plt.cm.plasma,
            hide_spines=True,
            colorbar=True,
            show_absolute=False,
            show_normed=True,
            class_names=['background']+list(LABEL2ORGAN.values()),
            fontcolor_threshold=100,
        )
        wandb.log({"Confusion Matrix": fig})

    def configure_optimizers(self):
        """
        Set optimizer and sheduler
        """
        optimizer = AdamW(params=self.trainer.model.parameters(), lr=self.learning_rate)
        sheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [sheduler]


if __name__ == '__main__':
    import segmentation_models_pytorch as smp
    model = LitSegmenter(
        net=smp.Unet(
            in_channels=1,
            classes=8,
        ),
        loss=smp.losses.DiceLoss(
            mode='multiclass'
        )
    )
    x = 0
