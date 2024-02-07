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
    def __init__(
        self,
        net: torch.nn.Module,
        loss: torch.nn.modules.loss._Loss,
        learning_rate: float = 3e-4,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['net', 'loss', 'learning_rate'])

        self.net: torch.nn.Module = net
        self.loss: torch.nn.modules.loss._Loss = loss

        # Need this variable for learning rate tuner
        self.learning_rate = learning_rate

        # Metrics
        self.val_dices: list[Metric] = [MeanMetric() for _ in LABEL2ORGAN.values()]

        # Variables for confusion Matrix
        self.confmat: Metric = ConfusionMatrix(task="multiclass", num_classes=len(LABEL2ORGAN)+1)
        self.matrix: torch.Tensor = torch.zeros((len(LABEL2ORGAN)+1, len(LABEL2ORGAN)+1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch

        pred = self.forward(x)
        loss = self.loss(pred, y)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def on_validation_epoch_start(self) -> None:
        for metric in self.val_dices:
            metric.reset()
            metric.to(self.device)
        self.matrix = torch.zeros((len(LABEL2ORGAN)+1, len(LABEL2ORGAN)+1), device=self.device)

    def validation_step(self, batch, batch_idx) -> None:
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
