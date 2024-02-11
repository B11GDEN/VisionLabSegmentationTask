from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import argparse

import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation.data.transforms import ClipAndNorm, PadWithDivisor, LABEL2ORGAN


def parse_args():
    parser = argparse.ArgumentParser(description="Predict")
    parser.add_argument("--model", type=str, help="# path to model weights")
    parser.add_argument("--src", type=str, help="# path with images scans")
    parser.add_argument("--dst", type=str, help="# path to save model result")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # config
    args = parse_args()

    # load model
    model = torch.load(args.model)
    model.eval()
    model.cuda()

    # define transform
    test_transform = A.Compose([
        ClipAndNorm(min_value=-350, max_value=400),
        PadWithDivisor(divisor=32, value=0),
        ToTensorV2(),
    ])

    # Paths
    src = Path(args.src)
    dst = Path(args.dst)

    # Inference Cycle
    for im in tqdm(src.glob('*.npy'), desc='Inference'):
        # load img
        img = np.load(im)
        height, width = img.shape

        # apply transform
        img = test_transform(image=img)['image']

        # forward
        raw_mask = model(img.unsqueeze(0).cuda())[0]

        # post-process
        raw_mask = raw_mask.argmax(dim=0)  # get classes
        raw_mask = raw_mask[:height, :width]  # because of PadWithDivisor transform
        raw_mask = raw_mask.cpu().numpy()

        # convert classes
        mask = np.zeros_like(raw_mask)
        for idx, cl in enumerate(LABEL2ORGAN.keys()):
            mask[raw_mask == (idx+1)] = cl

        # save_mask
        np.save(dst / im.name, mask)