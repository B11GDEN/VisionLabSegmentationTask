import numpy as np
from albumentations import DualTransform


class ClipAndNorm(DualTransform):

    def __init__(self, min_value: float = -350, max_value: float = 400, always_apply: bool = True, p: float = 1.0):
        super().__init__(always_apply, p)
        self.min_value = min_value
        self.max_value = max_value

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        img = np.clip(img, self.min_value, self.max_value)
        img = (img - self.min_value) / (self.max_value - self.min_value)
        return img

    def apply_to_mask(self, mask: np.ndarray, **params) -> np.ndarray:
        return mask

    def __repr__(self):
        return f'ClipAndNorm({self.min_value} {self.max_value})'
