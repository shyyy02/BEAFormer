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
import scipy.io as sio
import h5py
import random


def read_mat(mat_path):
    try:
        mat = sio.loadmat(mat_path)
        cjdata = mat["cjdata"]
        if isinstance(cjdata, np.ndarray) and cjdata.shape == (1, 1):
            cjdata = cjdata[0, 0]
        img = cjdata["image"]
        lbl = (
            int(cjdata["label"][0, 0])
            if cjdata["label"].size == 1
            else int(cjdata["label"][0])
        )
        msk = cjdata["tumorMask"]
        if isinstance(img, np.ndarray) and img.ndim == 0:
            img = img.item()
        if isinstance(msk, np.ndarray) and msk.ndim == 0:
            msk = msk.item()
        return np.squeeze(img).astype(np.float64), lbl, np.squeeze(msk).astype(np.uint8)
    except NotImplementedError:
        with h5py.File(mat_path, "r") as f:
            if "cjdata" in f:
                cj = f["cjdata"]
                img = cj["image"][()]
                lbl_data = cj["label"][()]
                lbl = int(lbl_data[0]) if lbl_data.size == 1 else int(lbl_data)
                msk = cj["tumorMask"][()]
            else:
                img = msk = None
                lbl = None
                for key in f.keys():
                    if "image" in f[key]:
                        img = f[key]["image"][()]
                    if "label" in f[key]:
                        ld = f[key]["label"][()]
                        lbl = int(ld[0]) if ld.size == 1 else int(ld)
                    if "tumorMask" in f[key]:
                        msk = f[key]["tumorMask"][()]
            img = np.squeeze(img).astype(np.float64)
            msk = np.squeeze(msk).astype(np.uint8)
            return img, lbl, msk


def preprocessing(
    src_root: str,
    dst_root: str = None,
    split_ratio: tuple | list = (0.7, 0.15, 0.15),
    resize: tuple | list = (224, 224),
    clip_upper: float = 1976.0, # 99% 分位数
):
    assert abs(sum(split_ratio) - 1.0) < 1e-6

    H, W = resize

    if dst_root is None:
        parent_dir = os.path.dirname(src_root.rstrip(os.sep))
        dst_root = tempfile.mkdtemp(prefix="bkai_tmp_", dir=parent_dir)
        move_mode = True
    else:
        move_mode = False

    mat_files = glob(os.path.join(src_root, "*.mat"))
    groups = {1: [], 2: [], 3: []}
    for f in tqdm(mat_files, desc="读取 *.mat"):
        img, lbl, mask = read_mat(f)
        groups[lbl].append((img, mask))

    random.seed(1234)
    np.random.seed(1234)

    splits = {"train": [], "valid": [], "test": []}
    for items in groups.values():
        n = len(items)
        n_train = int(round(n * split_ratio[0]))
        n_valid = int(round(n * split_ratio[1]))
        n_test = n - n_train - n_valid
        splits["train"].extend([(img, mask) for img, mask in items[:n_train]])
        splits["valid"].extend(
            [(img, mask) for img, mask in items[n_train : n_train + n_valid]]
        )
        splits["test"].extend([(img, mask) for img, mask in items[n_train + n_valid :]])

    for split in splits:
        random.shuffle(splits[split])

    def preprocess_img(img):
        img = np.clip(img, 0, clip_upper)
        mean, std = img.mean(), img.std()
        if std < 1e-6:
            std = 1.0
        img = (img - mean) / std
        img = cv2.resize(img, resize, interpolation=cv2.INTER_LINEAR)
        return np.stack([img] * 3, axis=-1).astype(np.float32)

    def preprocess_mask(mask):
        mask = (mask > 0).astype(np.uint8)
        return cv2.resize(mask, resize, interpolation=cv2.INTER_NEAREST).astype(
            np.uint8
        )

    global_idx = 0
    for split, data in splits.items():
        dst_img_dir = os.path.join(dst_root, split, "images")
        dst_lbl_dir = os.path.join(dst_root, split, "labels")
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        for img, mask in tqdm(data, desc=f"Preprocessing BTMRI {split} data"):
            np.save(
                os.path.join(dst_img_dir, f"{global_idx:04d}.npy"), preprocess_img(img)
            )
            np.save(
                os.path.join(dst_lbl_dir, f"{global_idx:04d}.npy"),
                preprocess_mask(mask),
            )
            global_idx += 1

    if move_mode:
        shutil.rmtree(src_root)
        os.rename(dst_root, src_root)

    print("Preprocessing completed!")


class BTMRI(Dataset):
    def __init__(self, root: str, split: str):
        assert split in ["train", "valid", "test"], f"Invalid split: {split}"

        self.split = split

        image_dir = os.path.join(root, self.split, "images")
        label_dir = os.path.join(root, self.split, "labels")

        self.image_fpaths = sorted(glob(os.path.join(image_dir, "*")))
        self.label_fpaths = sorted(glob(os.path.join(label_dir, "*")))

        logging.info(
            f"BTMRI loaded {len(self.image_fpaths)} samples for {self.split}"
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
        image = image.astype(np.float32)
        label = label.astype(np.uint8)

        if self.split == "train":
            trans = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
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
                    )
                ]
            )

            data = trans(image=image, mask=label)
            image = data["image"]
            label = data["mask"]

        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        return image.float(), label.long()
