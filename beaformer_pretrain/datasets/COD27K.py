import os
import shutil
from torch.utils.data import Dataset
import imageio.v2 as imageio
import numpy as np
import torch
from glob import glob
import logging
from tqdm import tqdm
import utils
import tempfile
import torchvision.transforms.functional as tf
import cv2
import albumentations as A
import sys

"""
Input file structure:
    COD27K/
    ├── ACD1K/
    │   ├── images
    │   │   ├── xxx.jpg
    │   │   └── ...
    │   └── labels
    │       ├── xxx.jpg
    │       └── ...
    ├── CAMO/
    ├── CHAMELEON/
    ├── CoCOD8K/
    ├── COD10K/
    ├── CPD1K/
    ├── NC4K/
    └── PlantCamo/
        ├── images
        │   ├── xxx.jpg
        │   └── ...
        └── labels
            ├── xxx.png
            └── ...
"""

def preprocessing(src_root: str, dst_root: str = None, resize: tuple | list = (224, 224)):
    H, W = resize

    if dst_root is None:
        parent_dir = os.path.dirname(src_root.rstrip(os.sep))
        dst_root = tempfile.mkdtemp(prefix="COD27K_tmp_", dir=parent_dir)
        move_mode = True
    else:
        move_mode = False

    datasets = os.listdir(src_root)
    dst_img_dir = os.path.join(dst_root, "images")
    dst_lbl_dir = os.path.join(dst_root, "labels")
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    global_idx = 0

    for i, dataset in enumerate(datasets, 1):
        src_img_dir = os.path.join(src_root, dataset, "images")
        src_lbl_dir = os.path.join(src_root, dataset, "labels")

        image_fpaths = sorted(glob(os.path.join(src_img_dir, "*")))
        label_fpaths = sorted(glob(os.path.join(src_lbl_dir, "*")))

        for src_img_fpath, src_lbl_fpath in tqdm(zip(image_fpaths, label_fpaths), desc=f"Preprocessing COD27K"):
            img_fname = os.path.splitext(os.path.basename(src_img_fpath))[0]
            lbl_fname = os.path.splitext(os.path.basename(src_lbl_fpath))[0]

            if img_fname != lbl_fname:
                print(f"Mismatching!\ndataset: {dataset}\nimage name: {img_fname}\nlabel name: {lbl_fname}")
                sys.exit(1)
        
            image = imageio.imread(src_img_fpath).astype(np.uint8)
            label = (imageio.imread(src_lbl_fpath) > 0).astype(np.uint8)

            image = cv2.resize(image, (H, W), interpolation=cv2.INTER_LINEAR)
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.ndim == 3:
                if image.shape[2] == 1:
                    image = np.repeat(image, 3, axis=2)
                elif image.shape[2] == 4:
                    image = image[:, :, :3]
        
            label = cv2.resize(label, (H, W), interpolation=cv2.INTER_NEAREST)
            if label.ndim == 3:
                label = label[..., 0]

            np.save(os.path.join(dst_img_dir, f"{global_idx:06d}.npy"), image)
            np.save(os.path.join(dst_lbl_dir, f"{global_idx:06d}.npy"), label)

            global_idx += 1
        
        print(f"[{i}/{len(datasets)}] dataset completed")

    if move_mode:
        shutil.rmtree(src_root)
        os.rename(dst_root, src_root)

    print("Preprocessing completed!")


class COD27K(Dataset):
    def __init__(self, root: str, use_dist_weight: bool = True):
        self.use_dist_weight = use_dist_weight

        image_dir = os.path.join(root, "images")
        label_dir = os.path.join(root, "labels")

        self.image_fpaths = sorted(glob(os.path.join(image_dir, "*")))
        self.label_fpaths = sorted(glob(os.path.join(label_dir, "*")))

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        logging.info(
            f"COD27K loaded {len(self.image_fpaths)} samples"
        )

    def __getitem__(self, idx: int):
        image = np.load(self.image_fpaths[idx])
        label = np.load(self.label_fpaths[idx])

        if image.shape[-1] != 1 and image.shape[-1] != 3:
            print(image.shape[-1])
            print(self.image_fpaths[idx])

        image, label = self.transform(image, label)
        bnd_label = utils.get_bnd_label(label)
        sample = {
            "image": image,
            "label": label,
            "bnd_label": bnd_label,
        }
        if self.use_dist_weight:
            sample["dist_weight"] = utils.get_dist_weight(bnd_label)
        return sample

    def __len__(self):
        return len(self.image_fpaths)

    def transform(self, image, label):
        image = image.astype(np.uint8)
        label = label.astype(np.uint8)

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
        image = tf.normalize(image, self.mean, self.std)
        label = torch.from_numpy(label)

        return image.float(), label.long()