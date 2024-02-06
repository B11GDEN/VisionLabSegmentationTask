import numpy as np
from albumentations import DualTransform

ORGAN2CLASS = {
    1: 1,  # селезенка
    2: 2,  # правая почка
    3: 3,  # левая почка
    6: 4,  # печень
    7: 5,  # желудок
    8: 6,  # аорта
    9: 7,  # поджелудочная железа
}


class ConvertMask(DualTransform):

    def __init__(self, always_apply: bool = True, p: float = 1.0):
        super().__init__(always_apply, p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return img

    def apply_to_mask(self, raw_mask: np.ndarray, **params) -> np.ndarray:
        mask = np.zeros_like(raw_mask, dtype=np.int64)
        for org, cl in ORGAN2CLASS.items():
            mask[raw_mask == org] = cl
        return mask

    def __repr__(self):
        return 'ConvertMask()'
