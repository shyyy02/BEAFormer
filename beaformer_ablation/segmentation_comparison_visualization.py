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
Candidates require good, similar Dice across all models, low nonboundary error,
and Ours better than every baseline in both Dice and GT-boundary-band error.
Rank by the smallest boundary error advantage, then Ours Dice.
Boundary band is dilation(GT)-erosion(GT) with a square radius in pixels.
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


def region_error_rates(pred, gt, radius):
    import cv2
    gt = np.asarray(gt, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError("Prediction/GT shapes differ")
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    mask = gt.astype(np.uint8)
    dilated = cv2.dilate(mask, kernel, borderType=cv2.BORDER_REPLICATE)
    eroded = cv2.erode(mask, kernel, borderType=cv2.BORDER_REPLICATE)
    band = dilated != eroded
    error = pred != gt
    return {"boundary_pixels": int(band.sum()), "nonboundary_pixels": int((~band).sum()),
            "boundary_error": float(error[band].mean()) if band.any() else None,
            "nonboundary_error": float(error[~band].mean()) if (~band).any() else None}


def candidate_metrics(row, min_dice, min_ours_dice, max_dice_spread, max_nonboundary_error):
    dices = [row[f"{name}_dice"] for name in MODEL_NAMES]
    dice_margins = [row["beaformer_dice"] - row[f"{name}_dice"] for name in MODEL_NAMES[:-1]]
    boundary_errors = [row[f"{name}_boundary_error"] for name in MODEL_NAMES]
    nonboundary_errors = [row[f"{name}_nonboundary_error"] for name in MODEL_NAMES]
    valid_regions = all(v is not None for v in boundary_errors + nonboundary_errors)
    boundary_margins = ([row[f"{name}_boundary_error"] - row["beaformer_boundary_error"]
                         for name in MODEL_NAMES[:-1]] if valid_regions else [])
    result = {"min_dice_advantage": min(dice_margins), "mean_dice_advantage": float(np.mean(dice_margins)),
              "min_model_dice": min(dices), "dice_spread": max(dices) - min(dices),
              "max_model_nonboundary_error": max(nonboundary_errors) if valid_regions else None,
              "min_boundary_advantage": min(boundary_margins) if valid_regions else None,
              "mean_boundary_advantage": float(np.mean(boundary_margins)) if valid_regions else None}
    reasons = []
    if row["foreground_pixels"] == 0:
        reasons.append("empty_roi")
    if min(dices) < min_dice:
        reasons.append("low_model_dice")
    if row["beaformer_dice"] < min_ours_dice:
        reasons.append("low_ours_dice")
    if result["dice_spread"] > max_dice_spread + 1e-12:
        reasons.append("large_dice_spread")
    if min(dice_margins) <= 0:
        reasons.append("ours_not_best_dice")
    if not valid_regions:
        reasons.append("empty_boundary_or_nonboundary")
    else:
        if result["max_model_nonboundary_error"] > max_nonboundary_error:
            reasons.append("large_nonboundary_error")
        if result["min_boundary_advantage"] <= 0:
            reasons.append("ours_not_best_boundary")
    return {**result, "eligible": not reasons, "rejection_reasons": ";".join(reasons)}


def select_candidates(rows, count):
    candidates = [row for row in rows if row["eligible"]]
    candidates.sort(key=lambda row: (-row["min_boundary_advantage"], -row["beaformer_dice"], row["dataset_index"]))
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
    parser.add_argument("--min-dice", type=float, default=0.85, help="Minimum Dice for every model")
    parser.add_argument("--min-ours-dice", type=float, default=0.90)
    parser.add_argument("--max-dice-spread", type=float, default=0.05, help="Maximum max(Dice)-min(Dice) across all six models")
    parser.add_argument("--max-nonboundary-error", type=float, default=0.01, help="Maximum nonboundary pixel error rate for every model")
    parser.add_argument("--boundary-band-radius", type=int, default=2, help="GT boundary-band radius in pixels; separate from contour drawing width")
    parser.add_argument("--alpha", type=float, default=0.4, help="Prediction/GT fill opacity, identical across models")
    parser.add_argument("--boundary-width", type=int, default=1, help="GT contour width in pixels")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if not 0 < args.alpha < 1 or args.boundary_width < 1 or args.num_candidates < 1:
        parser.error("Require 0<alpha<1, boundary-width>=1 and num-candidates>=1")
    if any(not 0 <= value <= 1 for value in (args.min_dice, args.min_ours_dice, args.max_dice_spread, args.max_nonboundary_error)):
        parser.error("Dice/error thresholds must be between 0 and 1")
    if args.boundary_band_radius < 1:
        parser.error("--boundary-band-radius must be >=1")
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
                    rates = region_error_rates(mask, label, args.boundary_band_radius)
                    rows[index][f"{name}_boundary_error"] = rates["boundary_error"]
                    rows[index][f"{name}_nonboundary_error"] = rates["nonboundary_error"]
                    rows[index]["boundary_pixels"] = rates["boundary_pixels"]
                    rows[index]["nonboundary_pixels"] = rates["nonboundary_pixels"]
                    rows[index]["foreground_pixels"] = int(label.sum())
                    Image.fromarray(mask.astype(np.uint8) * 255).save(cache / name / f"{index}.png")
            del model, result, logits
            torch.cuda.empty_cache()
            print(f"Evaluated {name}: {len(dataset)} test images", flush=True)
        for row in rows:
            row.update(candidate_metrics(row, args.min_dice, args.min_ours_dice,
                                         args.max_dice_spread, args.max_nonboundary_error))
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
                   "selection": "All model Dice >= min_dice; Ours Dice >= min_ours_dice and > every baseline; Dice spread <= max_dice_spread; every nonboundary error <= max_nonboundary_error; Ours boundary error < every baseline; nonempty ROI/boundary/nonboundary",
                   "ranking": "Minimum baseline boundary error minus Ours boundary error, descending; ties by Ours Dice descending and index ascending",
                   "boundary_band_definition": "dilation(GT)-erosion(GT), square kernel size 2*boundary_band_radius+1, replicated image borders",
                   "regional_error_definition": "Mean of (binary prediction != GT) inside/outside the GT boundary band",
                   "test_mean_dice": {name: float(np.mean([r[f"{name}_dice"] for r in rows])) for name in MODEL_NAMES},
                   "threshold": "logits > 0 for all models", "gt_color": "#2ECC71", "prediction_color": "#3B82F6",
                   "export": "Separate PNG/PDF panels; PNG preserves preprocessed resolution with 600 DPI metadata"}
        with open(output / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, allow_nan=False)
        if not selected:
            print("No sample satisfies the quality and boundary-advantage filters; inspect per_sample.csv rejection_reasons or adjust thresholds. No fallback examples exported.", flush=True)
    print(f"Exported {len(selected)} candidates to {output.resolve()}; inspect structure and boundaries for final selection.")


if __name__ == "__main__":
    main()
