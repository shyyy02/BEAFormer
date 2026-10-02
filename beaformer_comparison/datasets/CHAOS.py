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
import random
import pydicom

"""
Input file structure:
    CHAOS/
    ├── images/
    │   ├── 1
    │   │   ├── IMG-0004-00002.dcm
    |   │   └── ...
    │   └── ...
    └── labels/
        ├── 1
        │   ├── IMG-0004-00002.png
        │   └── ...
        └── ...
"""

def preprocessing(
    src_root: str,
    dst_root: str = None,
    resize: tuple | list = (224, 224),
):
    if dst_root is None:
        parent_dir = os.path.dirname(src_root.rstrip(os.sep))
        dst_root = tempfile.mkdtemp(prefix="chaos_tmp_", dir=parent_dir)
        move_mode = True
    else:
        move_mode = False
    
    H, W = resize

    lut = np.zeros(256, dtype=np.uint8)
    # label_map = {
    #     0: 0,       # background
    #     63: 1,      # Liver
    #     126: 2,     # Right kidney
    #     189: 3,     # Left kidney
    #     252: 4,     # Spleen
    # }
    label_map = {
        0: 0,       # background
        63: 0,      # Liver
        126: 1,     # Right kidney
        189: 1,     # Left kidney
        252: 0,     # Spleen
    }
    for k, v in label_map.items():
        lut[k] = v

    src_img_dir = os.path.join(src_root, "images")
    src_lbl_dir = os.path.join(src_root, "labels")
    dst_img_sli_dir = os.path.join(dst_root, "slices", "images")
    dst_lbl_sli_dir = os.path.join(dst_root, "slices", "labels")
    dst_img_vol_dir = os.path.join(dst_root, "volumes", "images")
    dst_lbl_vol_dir = os.path.join(dst_root, "volumes", "labels")

    for d in [dst_img_sli_dir, dst_lbl_sli_dir, dst_img_vol_dir, dst_lbl_vol_dir]:
        os.makedirs(d, exist_ok=True)

    case_ids = sorted(os.listdir(src_img_dir))

    for case_id in tqdm(case_ids, desc="Processing CHAOS data"):
        src_img_case_dir = os.path.join(src_img_dir, case_id)
        src_lbl_case_dir = os.path.join(src_lbl_dir, case_id)

        dst_img_case_dir = os.path.join(dst_img_sli_dir, case_id)
        dst_lbl_case_dir = os.path.join(dst_lbl_sli_dir, case_id)
        os.makedirs(dst_img_case_dir, exist_ok=True)
        os.makedirs(dst_lbl_case_dir, exist_ok=True)

        img_fnames = sorted(os.listdir(src_img_case_dir))
        lbl_fnames = sorted(os.listdir(src_lbl_case_dir))

        img_vol = []
        lbl_vol = []

        for idx, (img_fname, lbl_fname) in enumerate(zip(img_fnames, lbl_fnames)):

            image = pydicom.dcmread(os.path.join(src_img_case_dir, img_fname)).pixel_array.astype(np.float32)
            label = imageio.imread(os.path.join(src_lbl_case_dir, lbl_fname)).astype(np.uint8)
            label = lut[label]

            image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (W, H), interpolation=cv2.INTER_NEAREST)

            np.save(os.path.join(dst_img_case_dir, f"{idx:03d}.npy"), image)
            np.save(os.path.join(dst_lbl_case_dir, f"{idx:03d}.npy"), label)

            img_vol.append(image)
            lbl_vol.append(label)

        img_vol = np.stack(img_vol, axis=-1)
        lbl_vol = np.stack(lbl_vol, axis=-1)

        np.save(os.path.join(dst_img_vol_dir, f"{case_id}.npy"), img_vol)
        np.save(os.path.join(dst_lbl_vol_dir, f"{case_id}.npy"), lbl_vol)
    
    if move_mode:
        shutil.rmtree(src_root)
        os.rename(dst_root, src_root)

    print("Preprocessing completed!")



class CHAOS(Dataset):
    def __init__(self, root: tuple | list, case_ids: tuple | list, split: str):
        assert split in ["train", "valid"], f"Invalid split: {split}"
        
        self.split = split
        self.image_fpaths = []
        self.label_fpaths = []

        for case_id in case_ids:
            if split == "train":
                image_dir = os.path.join(root, f"slices/images/{case_id}")
                label_dir = os.path.join(root, f"slices/labels/{case_id}")
                self.image_fpaths.extend(sorted(glob(os.path.join(image_dir, "*"))))
                self.label_fpaths.extend(sorted(glob(os.path.join(label_dir, "*"))))
    
            else:
                self.image_fpaths.append(os.path.join(root, f"volumes/images/{case_id}.npy"))
                self.label_fpaths.append(os.path.join(root, f"volumes/labels/{case_id}.npy"))

        logging.info(
            f"CHAOS loaded {len(self.image_fpaths)}  {'slices' if self.split == 'train' else 'volumes'} for {self.split}"
        )

    def __getitem__(self, idx):

        image = np.load(self.image_fpaths[idx])
        label = np.load(self.label_fpaths[idx])

        if self.split == "train":
            image, label = self.transform_slice(image, label, augment=True)
            sample = {
                "image": image,
                "label": label,
            }

        else:
            images = []
            labels = []
            for i in range(image.shape[-1]):
                img, lbl = self.transform_slice(image[..., i], label[..., i], augment=False)
                images.append(img)
                labels.append(lbl)

            images = torch.stack(images)
            labels = torch.stack(labels)
            sample = {"image": images, "label": labels}
            
        return sample

    def __len__(self):
        return len(self.image_fpaths)

    def transform_slice(self, image, label, augment):
        image = image.astype(np.float32)
        image = (image - image.mean()) / (image.std() + 1e-8)
        image = np.expand_dims(image, axis=-1)
        label = label.astype(np.uint8)
        
        if augment:
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
                    )
                ]
            )

            data = trans(image=image, mask=label)
            image = data["image"]
            label = data["mask"]

        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        image = image.repeat(3, 1, 1)
        label = torch.from_numpy(label)

        return image.float(), label.long()
