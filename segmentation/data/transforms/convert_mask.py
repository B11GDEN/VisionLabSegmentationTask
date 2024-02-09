import numpy as np
from albumentations import DualTransform

LABEL2ORGAN = {
    1: 'spleen',  # селезенка
    2: 'right_kidney',  # правая почка
    3: 'left_kidney',  # левая почка
    6: 'liver',  # печень
    7: 'stomach',  # желудок
    8: 'aorta',  # аорта
    9: 'pancreas',  # поджелудочная железа
}


class ConvertMask(DualTransform):
    """
        Augmentations to translate source classes to range (0, N), where N is number of classes

        Methods:
            __init__: Initialize the `ConvertMask` object
            apply: Apply augmentation to img
            apply_to_mask: Apply augmentation to mask

    """
    def __init__(self, always_apply: bool = True, p: float = 1.0):
        super().__init__(always_apply, p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return img

    def apply_to_mask(self, raw_mask: np.ndarray, **params) -> np.ndarray:
        mask = np.zeros_like(raw_mask, dtype=np.int64)
        for idx, cl in enumerate(LABEL2ORGAN.keys()):
            mask[raw_mask == cl] = (idx+1)
        return mask

    def __repr__(self):
        return 'ConvertMask()'
