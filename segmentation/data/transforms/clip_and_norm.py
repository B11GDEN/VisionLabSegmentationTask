import numpy as np
from albumentations import DualTransform


class ClipAndNorm(DualTransform):
    """
        Augmentation to normalize KT images.
        First clipping the values and then converts them to a range (0, 1).

        Note:
            Default values (-350, 400) define the soft tissue range of the Hounsfield scale.

        Attributes:
            min_value (int): Min clipping value, Default to `-350`
            max_value (int): Max clipping value, Default to `400`

        Methods:
            __init__: Initialize the `ClipAndNorm` object
            apply: Apply augmentation to img
            apply_to_mask: Apply augmentation to mask

    """
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
