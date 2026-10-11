"""Select and export six-model segmentation comparisons for one dataset per run.

Example:
    python beaformer_ablation/segmentation_comparison_visualization.py \
        --dataset polypdb_wli --root /path/PolypDB-WLI_224x224 --gpu 0 \
        --unet /path/unet.pth --uctransnet /path/uctransnet.pth \
        --swinunet /path/swinunet.pth --pvtemcad /path/pvtemcad.pth \
        --pranet /path/pranet.pth --beaformer /path/ours.pth \
        --output-dir comparison_visuals/polypdb_wli

Repeat with dataset-specific root/checkpoints for busi, glas and btmri.
No config files are read. Use the existing test-set preprocessing and logits>0.
Candidates require a nonempty ROI and Ours Dice strictly above ALL five models.
Rank by the smallest Dice advantage over the five models, then Ours Dice.
Export the first --num-candidates for manual review; do not silently fall back.
All-test scores and candidate rankings are saved separately from selected images.
Each candidate exports eight standalone PNG/PDF panels: input, GT ROI overlay,
and six prediction overlays. GT is green; predictions are translucent blue;
every prediction panel has the same solid green GT contour. No merged figures.
PNG retains the preprocessed pixel size with 600 DPI metadata.
"""

import argparse
from contextlib import contextmanager
import csv
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types

import numpy as np
from PIL import Image
import torch


DATASETS = {"polypdb_wli": "PolypDB_WLI", "busi": "BUSI", "glas": "GlaS", "btmri": "BTMRI"}
MODEL_NAMES = ("unet", "uctransnet", "swinunet", "pvtemcad", "pranet", "beaformer")
GT_COLOR = np.array([46, 204, 113], dtype=np.float64)  # #2ECC71
PRED_COLOR = np.array([59, 130, 246], dtype=np.float64)  # #3B82F6


@contextmanager
def model_namespace(directory):
    """Temporarily resolve absolute 'models.*' imports in the intended project."""
    saved = {k: v for k, v in sys.modules.items() if k == "models" or k.startswith("models.")}
    for key in saved:
        del sys.modules[key]
    package = types.ModuleType("models")
    package.__path__ = [str(directory)]
    sys.modules["models"] = package
    try:
        yield
    finally:
        for key in list(sys.modules):
            if key == "models" or key.startswith("models."):
                del sys.modules[key]
        sys.modules.update(saved)


def build_model(name):
    project = Path(__file__).resolve().parent
    directory = project / "models" if name == "beaformer" else project.parent / "beaformer_comparison" / "models"
    with model_namespace(directory):
        if name == "beaformer":
            return importlib.import_module("models.BEAFormer").BEAFormer(num_classes=1)
        if name == "unet":
            return importlib.import_module("models.U_Net").U_Net(num_classes=1)
        if name == "uctransnet":
            return importlib.import_module("models.UCTransNet").UCTransNet(num_classes=1)
        if name == "swinunet":
            return importlib.import_module("models.SwinUnet").swinunet(num_classes=1, pretrained_backbone_fpath=None)
        if name == "pvtemcad":
            return importlib.import_module("models.PVT_EMCAD").EMCADNet(num_classes=1, pretrained_backbone_fpath=None)
        if name == "pranet":
            return importlib.import_module("models.PraNet").PraNet(pretrained_backbone_fpath=None)
    raise ValueError(f"Unknown model: {name}")


def load_checkpoint(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model_state_dict", "state_dict"):
        if key in state:
            state = state[key]
            break
    state = {k.removeprefix("module."): v for k, v in state.items()}
    # Comparison utils normally saves the wrapped model.net without its prefix.
    if state and all(k.startswith("net.") for k in state):
        state = {k[4:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def dice_score(pred, gt):
    denominator = int(pred.sum()) + int(gt.sum())
    return 2 * int((pred & gt).sum()) / denominator if denominator else 1.0


def select_candidates(rows, count):
    candidates = [row for row in rows if row["eligible"]]
    candidates.sort(key=lambda row: (-row["min_dice_advantage"], -row["beaformer_dice"], row["dataset_index"]))
    return [{**row, "rank": rank + 1, "exported": rank < count} for rank, row in enumerate(candidates)]


def input_rgb(image, dataset):
    values = image.cpu().permute(1, 2, 0).numpy().astype(np.float64)
    if hasattr(dataset, "mean") and hasattr(dataset, "std"):
        values = np.clip(values * np.asarray(dataset.std) + np.asarray(dataset.mean), 0, 1)
    else:
        values = (values - values.min()) / max(float(np.ptp(values)), 1e-8)
    return np.rint(values * 255).astype(np.uint8)


def overlay(rgb, mask, color, alpha):
    result = rgb.astype(np.float64).copy()
    result[mask] = (1 - alpha) * result[mask] + alpha * color
    return np.rint(result).astype(np.uint8)


def prediction_overlay(rgb, prediction, label, alpha, boundary_width):
    import cv2
    result = overlay(rgb, prediction, PRED_COLOR, alpha)
    # RETR_LIST includes inner ROI contours (holes) as well as outer contours.
    contours, _ = cv2.findContours(label.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(result, contours, -1, tuple(int(v) for v in GT_COLOR), thickness=boundary_width,
                     lineType=cv2.LINE_8)
    return result


def save_panel(path, pixels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Image.fromarray(pixels).save(path.with_suffix(".png"), dpi=(600, 600))
    h, w = pixels.shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(pixels, interpolation="nearest")
    ax.axis("off")
    fig.savefig(path.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def write_rows(path, rows, fields=None):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--root", required=True)
    for name in MODEL_NAMES:
        parser.add_argument(f"--{name}", required=True, help=f"{name} full checkpoint for the selected dataset")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-candidates", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.4, help="Prediction/GT fill opacity, identical across models")
    parser.add_argument("--boundary-width", type=int, default=1, help="GT contour width in pixels")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if not 0 < args.alpha < 1 or args.boundary_width < 1 or args.num_candidates < 1:
        parser.error("Require 0<alpha<1, boundary-width>=1 and num-candidates>=1")
    for name in MODEL_NAMES:
        if not Path(getattr(args, name)).is_file():
            parser.error(f"Checkpoint not found for {name}: {getattr(args, name)}")
    if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
        parser.error("--gpu must select an available CUDA device")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    name = DATASETS[args.dataset]
    dataset = getattr(importlib.import_module(f"datasets.{name}"), name)(root=args.root, split="test")
    if not len(dataset):
        parser.error("The test set is empty")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = [{"dataset_index": i, "source": str(dataset.image_fpaths[i])} for i in range(len(dataset))]
    with tempfile.TemporaryDirectory(prefix="prediction_cache_", dir=output) as temporary:
        cache = Path(temporary)
        for name in MODEL_NAMES:
            model = build_model(name)
            load_checkpoint(model, getattr(args, name))
            model = model.to(device).eval()
            (cache / name).mkdir()
            with torch.inference_mode():
                for index in range(len(dataset)):
                    sample = dataset[index]
                    label = sample["label"].cpu().numpy()
                    if label.ndim != 2 or not np.isin(label, [0, 1]).all():
                        raise ValueError("Expected binary GT [H,W]")
                    result = model(sample["image"].unsqueeze(0).to(device))
                    logits = result[0] if isinstance(result, (tuple, list)) else result
                    if logits.shape != (1, 1, *label.shape) or not torch.isfinite(logits).all():
                        raise ValueError(f"Invalid logits from {name}; expected [1,1,H,W]")
                    mask = (logits[0, 0] > 0).cpu().numpy()
                    rows[index][f"{name}_dice"] = dice_score(mask, label.astype(bool))
                    rows[index]["foreground_pixels"] = int(label.sum())
                    Image.fromarray(mask.astype(np.uint8) * 255).save(cache / name / f"{index}.png")
            del model, result, logits
            torch.cuda.empty_cache()
            print(f"Evaluated {name}: {len(dataset)} test images", flush=True)
        for row in rows:
            margins = [row["beaformer_dice"] - row[f"{name}_dice"] for name in MODEL_NAMES[:-1]]
            row["min_dice_advantage"] = min(margins)
            row["mean_dice_advantage"] = float(np.mean(margins))
            row["eligible"] = row["foreground_pixels"] > 0 and row["min_dice_advantage"] > 0
        ranked = select_candidates(rows, args.num_candidates)
        selected = [row for row in ranked if row["exported"]]
        write_rows(output / "per_sample.csv", rows)
        write_rows(output / "ranking.csv", ranked, fields=[*rows[0], "rank", "exported"])
        for row in selected:
            index = row["dataset_index"]
            sample = dataset[index]
            rgb = input_rgb(sample["image"], dataset)
            label = sample["label"].cpu().numpy().astype(bool)
            folder = output / f"rank_{row['rank']:03d}_sample_{index:05d}"
            folder.mkdir(parents=True, exist_ok=True)
            save_panel(folder / "input", rgb)
            save_panel(folder / "gt", overlay(rgb, label, GT_COLOR, args.alpha))
            for name in MODEL_NAMES:
                with Image.open(cache / name / f"{index}.png") as image:
                    pred = np.asarray(image).astype(bool)
                save_panel(folder / name, prediction_overlay(rgb, pred, label, args.alpha, args.boundary_width))
        summary = {"settings": vars(args), "num_test_images": len(dataset), "eligible_candidates": len(ranked),
                   "exported_candidates": len(selected), "selected_indices": [row["dataset_index"] for row in selected],
                   "selection": "Nonempty GT and Ours Dice > each baseline Dice; rank by minimum Dice margin, then Ours Dice",
                   "test_mean_dice": {name: float(np.mean([r[f"{name}_dice"] for r in rows])) for name in MODEL_NAMES},
                   "threshold": "logits > 0 for all models", "gt_color": "#2ECC71", "prediction_color": "#3B82F6",
                   "export": "Separate PNG/PDF panels; PNG preserves preprocessed resolution with 600 DPI metadata"}
        with open(output / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, allow_nan=False)
        if not selected:
            print("No sample satisfies Ours > all five baselines; no fallback examples exported.", flush=True)
    print(f"Exported {len(selected)} candidates to {output.resolve()}; inspect structure and boundaries for final selection.")


if __name__ == "__main__":
    main()
