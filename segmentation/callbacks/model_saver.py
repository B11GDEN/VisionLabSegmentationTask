from pathlib import Path
import torch
from lightning.pytorch.callbacks import ModelCheckpoint


class ModelSaver(ModelCheckpoint):
    """
    Override method on_save_checkpoint to save only net weights
    """
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        model_file = Path(self.best_model_path)
        save_dir = model_file.parent
        save_dir.mkdir(exist_ok=True, parents=True)
        (save_dir.parent / 'only_weights').mkdir(exist_ok=True)

        torch.save(pl_module.net, save_dir.parent / 'only_weights' / 'best.pt')