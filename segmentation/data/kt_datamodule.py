from pathlib import Path

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from segmentation.data.transforms import ClipAndNorm, ConvertMask


class KTDataset(Dataset):
    """
    Pytorch dataset for the VisionLabTask.

    Attributes:
        data_dir (Path): Root dataset directory
        files (list[str]): List of file names without extension
        transforms (A.Compose | None): Albumentations transformation, Defaults to `None`

    Methods:
        __init__: Initialize the `KTDataset` object
        __len__: Return number of files
        __getitem__: Return img and mask by index

    """
    def __init__(
            self,
            data_dir: Path,
            files: list[str],
            transforms: A.Compose | None = None,
    ) -> None:
        """
        Initialize the `KTDataset` object.

        Args:
            data_dir (Path): Root dataset directory
            files (list[str]): List of file names without extension
            transforms (A.Compose | None): Albumentations transformation, Defaults to `None`
        """
        super().__init__()

        self.data_dir: Path = data_dir
        self.files: list[str] = files

        self.transforms: A.Compose | None = transforms

    def __len__(self):
        """
        Returns:
            number of files
        """
        return len(self.files)

    def __getitem__(self, idx: int):
        """
        Takes an image and a mask by index, applies augmentations to them.

        Args:
            idx (int): Index of the file

        Returns:
            img (np.ndarray | torch.Tensor): augmented image.
            mask (np.ndarray | torch.Tensor): augmented mask.
        """
        img = np.load(str(self.data_dir / 'images' / f"{self.files[idx]}_img.npy"))
        mask = np.load(str(self.data_dir / 'labels' / f"{self.files[idx]}_lbl.npy"))

        if self.transforms:
            augment = self.transforms(image=img, mask=mask)
            img, mask = augment['image'], augment['mask']

        return img, mask


class KTDataModule(LightningDataModule):
    """
    `LightningDataModule` for the VisionLabTask dataset.

    The proposed data is obtained from the open datasets TotalSegmentator and AMOS22
    by sampling several coronal views from the 3D study array.

    Attributes:
        data_train (Dataset | None): Training dataset
        data_val (Dataset | None): Validation dataset
        data_test (Dataset | None): Test dataset

    Methods:
        __init__: Initialize the `KTDataModule` object
        setup: load data, Split to train val test
        train_dataloader: Create and return the train dataloader
        val_dataloader: Create and return the validation dataloader
        test_dataloader: Create and return the test dataloader

    """
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
        """
        Initialize a `KTDataModule`.

        Args:
            data_dir (str): Path to dataset directory
            train_transforms (A.Compose | None): Albumentations train transformation, Defaults to `None`
            val_transforms (A.Compose | None): Albumentations validation transformation, Defaults to `None`
            test_transforms (A.Compose | None): Albumentations test transformation, Defaults to `None`
            train_val_test_split (tuple[float, float, float]): The train, validation and test split.
                Defaults to `(0.8, 0.1, 0.1)`.
            batch_size (int): The batch size. Defaults to `64`.
            num_workers (int): The number of workers. Defaults to `0`.
            pin_memory (bool): Whether to pin memory. Defaults to `False`.
            seed (int): Random seed. Defaults to `42`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        Args:
            stage (str | None): The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
                Defaults to ``None``.
        """
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
        """
        Create and return the train dataloader.

        Returns:
            train_dataloader (DataLoader)
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create and return the validation dataloader.

        Returns:
            val_dataloader (DataLoader)
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create and return the test dataloader.

        Returns:
            test_dataloader (DataLoader)
        """
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
