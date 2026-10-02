import shutil
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf
import albumentations as A
import imageio.v2 as imageio
import os
from glob import glob
import numpy as np
import logging
from tqdm import tqdm
import tempfile
import utils
import cv2

"""
Input file structure:
    KvasirSEG/
    ├── train/
    │   ├── images
    │   └── labels
    └── test/
        ├── images
        └── labels
"""

def preprocessing(
    src_root: str,
    dst_root: str = None,
    resize: tuple | list = (224, 224),
):
    if dst_root is None:
        parent_dir = os.path.dirname(src_root.rstrip(os.sep))
        dst_root = tempfile.mkdtemp(prefix="KvasirSEG_tmp_", dir=parent_dir)
        move_mode = True
    else:
        move_mode = False
    
    H, W = resize

    global_idx = 0

    for split in ["train", "test"]:
        src_img_dir = os.path.join(src_root, split, "images")
        src_lbl_dir = os.path.join(src_root, split, "labels")
        dst_img_dir = os.path.join(dst_root, split, "images")
        dst_lbl_dir = os.path.join(dst_root, split, "labels")
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        img_fpaths = sorted(glob(os.path.join(src_img_dir, "*")))
        lbl_fpaths = sorted(glob(os.path.join(src_lbl_dir, "*")))

        for img_fpath, lbl_fpath in tqdm(zip(img_fpaths, lbl_fpaths), total=len(img_fpaths), desc=f"Preprocessing KvasirSEG {split} data"):
            image = imageio.imread(img_fpath).astype(np.uint8)
            label = (imageio.imread(lbl_fpath) > 0).astype(np.uint8)

            image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.ndim == 3 and image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)

            label = cv2.resize(label, (W, H), interpolation=cv2.INTER_NEAREST)
            if label.ndim == 3:
                label = label[..., 0]
            
            np.save(os.path.join(dst_img_dir, f"{global_idx:04d}.npy"), image)
            np.save(os.path.join(dst_lbl_dir, f"{global_idx:04d}.npy"), label)

            global_idx += 1

    if move_mode:
        shutil.rmtree(src_root)
        os.rename(dst_root, src_root)

    print("Preprocessing completed!")


class KvasirSEG(Dataset):
    def __init__(self, root: str, split: str):
        assert split in ["train", "test"], f"Invalid split: {split}"

        self.split = split
        
        image_dir = os.path.join(root, self.split, "images")
        label_dir = os.path.join(root, self.split, "labels")

        self.image_fpaths = sorted(glob(os.path.join(image_dir, "*")))
        self.label_fpaths = sorted(glob(os.path.join(label_dir, "*")))

        logging.info(
            f"KvasirSEG loaded {len(self.image_fpaths)} samples for {self.split}"
        )

    def __getitem__(self, idx: int):
        image = np.load(self.image_fpaths[idx])
        label = np.load(self.label_fpaths[idx])

        image, label = self.transform(image, label)
        sample = {
            "image": image,
            "label": label,
        }
        return sample

    def __len__(self):
        return len(self.image_fpaths)

    def transform(self, image, label):
        image = image.astype(np.uint8)
        label = label.astype(np.uint8)

        if self.split == "train":
            trans = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Affine(
                        translate_percent=(-0.05, 0.05),
                        rotate=(-15, 15),
                        scale=(0.9, 1.1),
                        interpolation=1,
                        mask_interpolation=0,
                        p=0.5,
                    ),
                    A.OneOf(
                        [
                            A.GridDistortion(
                                num_steps=5,
                                distort_limit=0.1,
                                interpolation=1,
                                mask_interpolation=0,
                            ),
                            A.ElasticTransform(
                                alpha=1,
                                sigma=50,
                                interpolation=1,
                                mask_interpolation=0,
                            ),
                        ],
                        p=0.3,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.15, contrast_limit=0.15, p=0.5
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=20,
                        val_shift_limit=10,
                        p=0.5,
                    ),
                    A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
                ]
            )

            data = trans(image=image, mask=label)
            image = data["image"]
            label = data["mask"]
    

        image = tf.to_tensor(image)
        label = torch.from_numpy(label)

        return image.float(), label.long()
