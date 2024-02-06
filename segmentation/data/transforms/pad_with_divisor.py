import numpy as np
from math import ceil
from albumentations import DualTransform


class PadWithDivisor(DualTransform):

    def __init__(self, divisor: int = 32, value: float = 0, always_apply: bool = True, p: float = 1.0):
        super().__init__(always_apply, p)
        self.divisor = divisor
        self.value = value

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        height, width = img.shape
        new_height, new_width = ceil(height / self.divisor) * self.divisor, ceil(width / self.divisor) * self.divisor
        img = np.pad(img, ((0, new_height-height), (0, new_width-width)), 'constant', constant_values=self.value)
        return img

    def apply_to_mask(self, mask: np.ndarray, **params) -> np.ndarray:
        height, width = mask.shape
        new_height, new_width = ceil(height / self.divisor) * self.divisor, ceil(width / self.divisor) * self.divisor
        mask = np.pad(mask, ((0, new_height - height), (0, new_width - width)), 'constant', constant_values=self.value)
        return mask

    def __repr__(self):
        return f'PadWithDivisor(divisor={self.divisor}, value={self.value})'
