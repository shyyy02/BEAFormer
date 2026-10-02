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
    BUSBRA/
    ├── images
    │   ├── xxx.jpeg
    │   └── ...
    └── labels
        ├── xxx.jpeg
        └── ...
"""

def preprocessing(
    src_root: str,
    dst_root: str = None,
    split_ratio: tuple | list = (0.7, 0.15, 0.15),
    resize: tuple | list = (224, 224),
):
    assert abs(sum(split_ratio) - 1.0) < 1e-6

    H, W = resize

    if dst_root is None:
        parent_dir = os.path.dirname(src_root.rstrip(os.sep))
        dst_root = tempfile.mkdtemp(prefix="BUSBRA_tmp_", dir=parent_dir)
        move_mode = True
    else:
        move_mode = False

    src_img_dir = os.path.join(src_root, "images")
    src_lbl_dir = os.path.join(src_root, "labels")

    img_fpaths = sorted(glob(os.path.join(src_img_dir, "*")))
    lbl_fpaths = sorted(glob(os.path.join(src_lbl_dir, "*")))

    np.random.seed(1234)
    ids = np.arange(len(img_fpaths))
    np.random.shuffle(ids)
    img_fpaths = [img_fpaths[i] for i in ids]
    lbl_fpaths = [lbl_fpaths[i] for i in ids]

    n_total = len(img_fpaths)
    n_train = int(n_total * split_ratio[0])
    n_valid = int(n_total * split_ratio[1])

    splits = {
        "train": {"images": img_fpaths[:n_train], "labels": lbl_fpaths[:n_train]},
        "valid": {"images": img_fpaths[n_train:n_train + n_valid], "labels": lbl_fpaths[n_train:n_train + n_valid]},
        "test": {"images": img_fpaths[n_train + n_valid:], "labels": lbl_fpaths[n_train + n_valid:]},
    }

    for split in ["train", "valid", "test"]:
        dst_img_dir = os.path.join(dst_root, split, "images")
        dst_lbl_dir = os.path.join(dst_root, split, "labels")
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        img_fpaths = splits[split]["images"]
        lbl_fpaths = splits[split]["labels"]

        for idx, (img_fpath, lbl_fpath) in enumerate(
            tqdm(
                zip(img_fpaths, lbl_fpaths),
                total=len(img_fpaths),
                desc=f"Preprocessing BUSBRA {split} data",
            )
        ):
            image = imageio.imread(img_fpath).astype(np.uint8)
            label = imageio.imread(lbl_fpath).astype(np.uint8)

            image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.ndim == 3 and image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)

            label = cv2.resize(label, (W, H), interpolation=cv2.INTER_NEAREST)
            
            
            np.save(os.path.join(dst_img_dir, f"{idx:04d}.npy"), image)
            np.save(os.path.join(dst_lbl_dir, f"{idx:04d}.npy"), label)

    if move_mode:
        shutil.rmtree(src_root)
        os.rename(dst_root, src_root)

    print("Preprocessing completed!")


class BUSBRA(Dataset):
    def __init__(self, root: str, split: str):
        assert split in ["train", "valid", "test"], f"Invalid split: {split}"

        self.split = split

        image_dir = os.path.join(root, self.split, "images")
        label_dir = os.path.join(root, self.split, "labels")

        self.image_fpaths = sorted(glob(os.path.join(image_dir, "*")))
        self.label_fpaths = sorted(glob(os.path.join(label_dir, "*")))

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        logging.info(
            f"BUSBRA loaded {len(self.image_fpaths)} samples for {self.split}"
        )

    def __getitem__(self, idx: int):
        image = np.load(self.image_fpaths[idx])
        label = np.load(self.label_fpaths[idx])

        image, label = self.transform(image, label)
        bnd_label = utils.get_bnd_label(label)
        sample = {
            "image": image,
            "label": label,
            "bnd_label": bnd_label,
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
