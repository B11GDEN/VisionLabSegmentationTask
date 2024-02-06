from pathlib import Path

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from segmentation.data.transforms import ClipAndNorm, ConvertMask


class KTDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        files: list[str],
        transforms: A.Compose | None = None,
    ) -> None:
        super().__init__()

        self.data_dir: Path = data_dir
        self.files: list[str] = files

        self.transforms: A.Compose | None = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        img = np.load(str(self.data_dir / 'images' / f"{self.files[idx]}_img.npy"))
        mask = np.load(str(self.data_dir / 'labels' / f"{self.files[idx]}_lbl.npy"))

        if self.transforms:
            augment = self.transforms(image=img, mask=mask)
            img, mask = augment['image'], augment['mask']

        return img, mask


class KTDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        train_transforms: A.Compose | None = None,
        val_transforms: A.Compose | None = None,
        test_transforms: A.Compose | None = None,
        train_val_test_split: tuple[float, float, float] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:

            data_dir = Path(self.hparams.data_dir)

            # get all filenames
            all_files = []
            for im in (data_dir / 'images').glob('*.npy'):
                all_files.append(im.name.split('_img')[0])

            # split files to train, val, test
            train_size, val_size, test_size = self.hparams.train_val_test_split
            train_files, val_test_files = train_test_split(
                all_files, test_size=val_size + test_size, random_state=self.hparams.seed)
            val_files, test_files = train_test_split(
                val_test_files, test_size=test_size / (val_size + test_size), random_state=self.hparams.seed)

            # Datasets
            self.data_train = KTDataset(
                data_dir=data_dir,
                files=train_files,
                transforms=self.hparams.train_transforms,
            )

            self.data_val = KTDataset(
                data_dir=data_dir,
                files=val_files,
                transforms=self.hparams.val_transforms,
            )

            self.data_test = KTDataset(
                data_dir=data_dir,
                files=test_files,
                transforms=self.hparams.test_transforms,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    size = 512
    datamodule = KTDataModule(
        data_dir="./dataset",
        train_transforms=A.Compose([
            A.PadIfNeeded(min_height=size, min_width=size, ),
            A.CropNonEmptyMaskIfExists(height=size, width=size),
            ClipAndNorm(),
            ConvertMask(),
            ToTensorV2(),
        ])
    )
    datamodule.setup()
    img, mask = datamodule.data_train[0]
    x = 0
