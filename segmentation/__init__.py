from nip import nip

from segmentation import data
from segmentation import lit_module
from segmentation.data import transforms

import albumentations

import segmentation_models_pytorch

nip(data)
nip(lit_module)
nip(transforms)

nip(albumentations)
nip(albumentations.pytorch)

nip(segmentation_models_pytorch)
nip(segmentation_models_pytorch.losses)