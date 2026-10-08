"""Export 10 random COD27K image/mask/boundary triplets as >=300 DPI PNGs.

python beaformer_pretrain/tmp.py --root /path/COD27K_224x224 --output-dir /path/examples

Supports preprocessed images/*.npy + labels/*.npy and original COD27K
subdataset/images + subdataset/labels folders. No augmentation or resizing.
PNG DPI metadata defaults to 600; original pixel resolution is preserved.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


EXTENSIONS = {".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def index_files(folder):
    files = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        if path.stem in files:
            raise ValueError(f"Duplicate sample stem in {folder}: {path.stem}")
        files[path.stem] = path
    return files


def find_pairs(root):
    if not root.is_dir():
        raise ValueError(f"Dataset root does not exist: {root}")
    folders = [root] if (root / "images").is_dir() else sorted(p for p in root.iterdir() if p.is_dir())
    pairs = []
    for folder in folders:
        if not (folder / "images").is_dir() or not (folder / "labels").is_dir():
            continue
        images = index_files(folder / "images")
        labels = index_files(folder / "labels")
        if images.keys() != labels.keys():
            raise ValueError(f"Unpaired filenames in {folder}: missing labels={sorted(images.keys()-labels.keys())[:5]}, missing images={sorted(labels.keys()-images.keys())[:5]}")
        pairs.extend((images[stem], labels[stem]) for stem in sorted(images))
    if len(pairs) < 10:
        raise ValueError(f"Need at least 10 paired samples; found {len(pairs)} in {root}")
    return pairs


def read_array(path):
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    with Image.open(path) as image:
        if image.mode == "P":
            image = image.convert("RGB")
        return np.asarray(image).copy()


def prepare_sample(image_path, label_path):
    image, label = read_array(image_path), read_array(label_path)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[-1] not in (3, 4)):
        raise ValueError(f"Unsupported image shape {image.shape}: {image_path}")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 255:
        raise ValueError(f"Expected unnormalized image values in [0,255]: {image_path}")
    if np.issubdtype(image.dtype, np.floating) and image.max() <= 1:
        image = image * 255
    image = np.rint(image).astype(np.uint8)
    if label.ndim == 3:
        label = label[..., 0]  # Same channel selection as COD27K preprocessing.
    if label.ndim != 2 or image.shape[:2] != label.shape:
        raise ValueError(f"Image/mask shape mismatch: {image_path}, {label_path}")
    if not np.isfinite(label).all():
        raise ValueError(f"Non-finite mask values: {label_path}")
    return image, (label > 0).astype(np.uint8)


def boundary_mask(label):
    """Matches utils.get_bnd_label(kernel_size=5), including both contour sides."""
    tensor = torch.from_numpy(label.astype(np.int64))
    boundary = torch.zeros_like(tensor)
    for value in torch.unique(tensor):
        mask = (tensor == value).float()[None, None]
        erosion = -F.max_pool2d(-mask, kernel_size=5, stride=1, padding=2)
        boundary = torch.maximum(boundary, (mask - erosion)[0, 0].long())
    return boundary.numpy().astype(np.uint8)


def export_samples(root, output_dir, seed=1234, dpi=600):
    if dpi < 300:
        raise ValueError("DPI must be at least 300")
    pairs = find_pairs(Path(root))
    selected = random.Random(seed).sample(pairs, 10)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (image_path, label_path) in enumerate(selected, 1):
        image, label = prepare_sample(image_path, label_path)
        boundary = boundary_mask(label)
        stem = f"sample_{index:02d}_{image_path.stem}"
        paths = {"image": f"{stem}_image.png", "mask": f"{stem}_mask.png", "boundary": f"{stem}_boundary.png"}
        for kind, array in (("image", image), ("mask", label * 255), ("boundary", boundary * 255)):
            Image.fromarray(array).save(output_dir / paths[kind], dpi=(dpi, dpi))
        records.append({"source_image": str(image_path.resolve()), "source_mask": str(label_path.resolve()),
                        "outputs": paths, "height": label.shape[0], "width": label.shape[1]})
        print(f"Exported {index}/10: {image_path}", flush=True)
    with open(output_dir / "samples.json", "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "dpi": dpi, "num_samples": 10, "samples": records}, f, indent=2, ensure_ascii=False)
    print(f"Saved 30 PNGs at {dpi} DPI to {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="/mnt/folder_3/shy/data/COD27K_224x224")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if args.dpi < 300:
        parser.error("--dpi must be >=300")
    export_samples(args.root, args.output_dir, args.seed, args.dpi)


if __name__ == "__main__":
    main()
