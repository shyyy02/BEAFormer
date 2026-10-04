"""Regional error increases relative to Normal under all-block interventions.

    python beaformer_ablation/regional_error_attribution.py --dataset polypdb_wli \
        --root /path/PolypDB-WLI_224x224 --checkpoint /path/best_model.pth \
        --gpu 0 --output-dir regional_error_results --top-fraction 0.2

Command-line settings only. All-En forces g=0 at every BEA block; Reverse
replaces g with 1-g at every block. P denotes sigmoid foreground probability.
A_Normal(mode) = abs(P_mode-Y) - abs(P_Normal-Y). Positive values mean Normal
is better; negative values mean the intervention is better. No clipping to positive
values or per-image color normalization is applied. Regions are disjoint:
Boundary=get_bnd_label(Y), Foreground Interior=(Y=1)&~Boundary,
Background=(Y=0)&~Boundary. The boundary includes both sides of the contour.
Statistics include pixel-pooled and image-wise macro means over the full test
set, with per-image results in CSV. Macro means exclude images where the
corresponding region is empty and report valid/excluded image counts.
Figures show Input | GT | Normal probability | Mode probability | Normal Advantage Map.
Existing delta_error statistics denote the same A_Normal quantity.
Candidates are selected separately for All-En and Reverse: require Dice_Normal
- Dice_mode > 0, boundary_mean_delta_error > 0 and S_boundary > 0,
then rank by S_boundary.
S_boundary = boundary mean A_Normal - max(interior mean A_Normal,
background mean A_Normal). Samples missing any region are not rankable.
Export the top 10%-20%
of filtered, rankable candidates for manual inspection of structural clarity.
Full-test-set statistics are not restricted to these selected examples.
"""

import argparse
import importlib
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from expert_divergence import DATASETS, load_model
from routing_intervention_eval import RoutingIntervention, write_csv


MODES = ("All-En", "Reverse")
REGIONS = ("boundary", "foreground_interior", "background")


def binary_dice(probability, label):
    pred, gt = probability > 0.5, label == 1
    denominator = int(pred.sum()) + int(gt.sum())
    return 2 * int((pred & gt).sum()) / denominator if denominator else 1.0


def selection_metrics(normal, prediction, label, stats):
    normal_dice = binary_dice(normal, label)
    mode_dice = binary_dice(prediction, label)
    boundary = stats["boundary_mean_delta_error"]
    ni, ng = stats["foreground_interior_pixels"], stats["background_pixels"]
    nonboundary = ((stats["foreground_interior_delta_error_sum"] + stats["background_delta_error_sum"])
                   / (ni + ng)) if ni + ng else None
    values = [stats[f"{r}_mean_delta_error"] for r in ("foreground_interior", "background")]
    score = boundary - max(values) if boundary is not None and all(v is not None for v in values) else None
    delta_dice = normal_dice - mode_dice
    rankable = score is not None and bool(np.isfinite(score))
    return {"normal_dice": normal_dice, "mode_dice": mode_dice, "delta_dice": delta_dice,
            "nonboundary_mean_delta_error": nonboundary, "S_boundary": score,
            "passes_filters": delta_dice > 0 and boundary is not None and boundary > 0 and rankable and score > 0,
            "rankable": rankable}


def rank_candidates(rows, mode, top_fraction):
    eligible = [r for r in rows if r["mode"] == mode and r["passes_filters"] and r["rankable"]]
    eligible.sort(key=lambda r: (-r["S_boundary"], r["dataset_index"]))
    n = math.ceil(len(eligible) * top_fraction)
    return [{**row, "rank": i + 1, "exported_for_review": i < n}
            for i, row in enumerate(eligible)]


def regional_counts(normal, prediction, label, boundary):
    if not (normal.shape == prediction.shape == label.shape == boundary.shape):
        raise ValueError("Probabilities, GT and boundary must have equal shapes")
    if not np.isin(label, [0, 1]).all():
        raise ValueError("GT must be binary")
    boundary = np.asarray(boundary, dtype=bool)
    masks = {"boundary": boundary, "foreground_interior": (label == 1) & ~boundary,
             "background": (label == 0) & ~boundary}
    error_normal = np.abs(normal.astype(np.float64) - label)
    error_mode = np.abs(prediction.astype(np.float64) - label)
    normal_advantage = error_mode - error_normal
    counts = {}
    for region, mask in masks.items():
        counts[f"{region}_pixels"] = int(mask.sum())
        counts[f"{region}_normal_error_sum"] = float(error_normal[mask].sum())
        counts[f"{region}_mode_error_sum"] = float(error_mode[mask].sum())
        counts[f"{region}_delta_error_sum"] = float(normal_advantage[mask].sum())
    return normal_advantage, counts


def regional_summary(counts):
    result = dict(counts)
    for region in REGIONS:
        n = counts[f"{region}_pixels"]
        for kind in ("normal_error", "mode_error", "delta_error"):
            result[f"{region}_mean_{kind}"] = counts[f"{region}_{kind}_sum"] / n if n else None
    return result


def image_macro_summary(rows):
    """Equal image weighting per region, using all evaluated rows before filtering."""
    summary = []
    for mode in MODES:
        mode_rows = [row for row in rows if row["mode"] == mode]
        for region in REGIONS:
            valid = [row for row in mode_rows if row[f"{region}_pixels"] > 0]
            result = {"mode": mode, "region": region,
                      "valid_images": len(valid), "excluded_empty_region_images": len(mode_rows) - len(valid),
                      "total_images": len(mode_rows)}
            for column, kind in (("mean_normal_error", "normal_error"),
                                 ("mean_mode_error", "mode_error"),
                                 ("mean_error_increase", "delta_error")):
                result[column] = float(np.mean([row[f"{region}_mean_{kind}"] for row in valid])) if valid else None
            summary.append(result)
    return summary


def dataset_boundary_scores(summary_rows, aggregation):
    """Apply the score formula to dataset-level region means, not image scores."""
    scores = []
    for mode in MODES:
        means = {row["region"]: row["mean_error_increase"]
                 for row in summary_rows if row["mode"] == mode}
        b, i, bg = (means[region] for region in REGIONS)
        score = b - max(i, bg) if all(v is not None for v in (b, i, bg)) else None
        scores.append({"mode": mode, "aggregation": aggregation,
                       "boundary_mean_delta_error": b, "interior_mean_delta_error": i,
                       "background_mean_delta_error": bg, "S_B": score})
    return scores


def plot_normal_advantage(path, image, label, normal, prediction, normal_advantage, mode, dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rgb = image.cpu().permute(1, 2, 0).numpy()
    if hasattr(dataset, "mean") and hasattr(dataset, "std"):
        rgb = np.clip(rgb * np.asarray(dataset.std) + np.asarray(dataset.mean), 0, 1)
    else:
        rgb = (rgb - rgb.min()) / max(float(np.ptp(rgb)), 1e-8)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), constrained_layout=True)
    axes[0].imshow(rgb)
    for ax, values in zip(axes[1:4], (label, normal, prediction)):
        ax.imshow(values, cmap="gray", vmin=0, vmax=1)
    heat = axes[4].imshow(normal_advantage, cmap="RdBu_r", vmin=-1, vmax=1)
    for ax, title in zip(axes, ("Input", "GT", "Normal (probability)",
                                f"{mode} (probability)", f"Normal Advantage over {mode}")):
        ax.set_title(title)
        ax.axis("off")
    colorbar = fig.colorbar(heat, ax=axes[4], shrink=0.8, ticks=[-1, 1])
    colorbar.ax.set_yticklabels([f"{mode} better", "Normal better"])
    path = Path(path)
    fig.savefig(
        path.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        path.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", default="pvt_v2_b2")
    parser.add_argument("--gpu", type=int, default=0, help="Index among visible CUDA devices")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-fraction", type=float, default=0.2,
                        help="Export the top 10%%-20%% of filtered candidates per mode (default: 0.2)")
    args = parser.parse_args()
    if not 0.1 <= args.top_fraction <= 0.2:
        parser.error("--top-fraction must be between 0.1 and 0.2")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        parser.error("--gpu must select an available CUDA device")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    from utils import get_bnd_label
    name = DATASETS[args.dataset]
    dataset = getattr(importlib.import_module(f"datasets.{name}"), name)(root=args.root, split="test")
    if not len(dataset):
        parser.error("The test set is empty")
    model = load_model(args.checkpoint, num_classes=1, encoder=args.encoder, device=device)
    intervention = RoutingIntervention(model)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    totals = {mode: {} for mode in MODES}
    rows = []
    review_summary = {}
    cache = tempfile.TemporaryDirectory(prefix="candidate_cache_", dir=output)
    try:
        with torch.inference_mode():
            for index in range(len(dataset)):
                sample = dataset[index]
                image, label_tensor = sample["image"], sample["label"].cpu()
                label = label_tensor.numpy()
                boundary = get_bnd_label(label_tensor).numpy().astype(bool)
                batch = image.unsqueeze(0).to(device)
                normal = intervention.predict(model, batch, "Normal", probabilities=True)[0].numpy()
                for mode in MODES:
                    prediction = intervention.predict(model, batch, mode, probabilities=True)[0].numpy()
                    normal_advantage, counts = regional_counts(normal, prediction, label, boundary)
                    for key, value in counts.items():
                        totals[mode][key] = totals[mode].get(key, 0) + value
                    stats = regional_summary(counts)
                    row = {"dataset_index": index, "source": str(dataset.image_fpaths[index]),
                           "mode": mode, **stats, **selection_metrics(normal, prediction, label, stats)}
                    rows.append(row)
                    if row["passes_filters"] and row["rankable"]:
                        key = mode.replace("-", "_")
                        np.savez_compressed(Path(cache.name) / f"{index}_{key}.npz", label=label,
                                            boundary=boundary, P_Normal=normal, P_mode=prediction,
                                            normal_advantage=normal_advantage)
                print(f"Evaluated {index + 1}/{len(dataset)}", flush=True)
        for mode in MODES:
            key = mode.replace("-", "_")
            ranked = rank_candidates(rows, mode, args.top_fraction)
            selected = [r for r in ranked if r["exported_for_review"]]
            review_summary[mode] = {"filtered_candidates": sum(r["mode"] == mode and r["passes_filters"] for r in rows),
                                    "rankable_candidates": len(ranked), "exported_candidates": len(selected),
                                    "dataset_indices": [r["dataset_index"] for r in selected]}
            folder = output / "candidates" / key
            folder.mkdir(parents=True, exist_ok=True)
            if ranked:
                write_csv(folder / "ranking.csv", ranked)
            else:
                (folder / "ranking.csv").write_text("dataset_index,mode,rank,S_boundary,delta_dice,exported_for_review\n", encoding="utf-8")
                print(f"{mode}: no candidates meet the criteria; no fallback samples exported", flush=True)
            for row in selected:
                index = row["dataset_index"]
                stem = f"rank_{row['rank']:04d}_sample_{index:05d}"
                with np.load(Path(cache.name) / f"{index}_{key}.npz") as arrays:
                    plot_normal_advantage(folder / f"{stem}.png", dataset[index]["image"], arrays["label"],
                                          arrays["P_Normal"], arrays["P_mode"], arrays["normal_advantage"], mode, dataset)
                    np.savez_compressed(folder / f"{stem}.npz", **dict(arrays))
    finally:
        intervention.close()
        cache.cleanup()

    summary_rows = []
    for mode in MODES:
        values = regional_summary(totals[mode])
        for region in REGIONS:
            summary_rows.append({"mode": mode, "region": region,
                                 "pixels": values[f"{region}_pixels"],
                                 "mean_normal_error": values[f"{region}_mean_normal_error"],
                                 "mean_mode_error": values[f"{region}_mean_mode_error"],
                                 "mean_error_increase": values[f"{region}_mean_delta_error"]})
    macro_rows = image_macro_summary(rows)
    dataset_scores = (dataset_boundary_scores(summary_rows, "pixel_pooled")
                      + dataset_boundary_scores(macro_rows, "image_macro"))
    for result_rows, aggregation in ((summary_rows, "pixel_pooled"), (macro_rows, "image_macro")):
        mode_scores = {row["mode"]: row["S_B"] for row in dataset_scores if row["aggregation"] == aggregation}
        for row in result_rows:
            row["S_B"] = mode_scores[row["mode"]]
    summary = {"settings": vars(args), "num_samples": len(dataset),
               "num_blocks": len(intervention.block_paths), "block_paths": intervention.block_paths,
               "definition": "A_Normal = abs(P_mode-Y)-abs(P_Normal-Y); P is sigmoid foreground probability; positive means Normal better, negative means intervention better",
               "map_name": "Normal Advantage Map",
               "statistics_naming": "delta_error and mean_error_increase fields measure the same A_Normal; retained for compatibility",
               "regions": {"boundary": "get_bnd_label(Y), both sides of contour",
                           "foreground_interior": "(Y==1) & ~boundary", "background": "(Y==0) & ~boundary"},
               "aggregation": "results: pixel-pooled; results_macro: equal-weight image-wise region means; both use all test images before visual filtering",
               "macro_empty_region_policy": "Exclude images with an empty corresponding region; report valid and excluded image counts; no valid images yields null means",
               "visual_selection": {"delta_dice": "Dice_Normal - Dice_mode; masks use P > 0.5",
                                    "filters": "delta_dice > 0 and boundary_mean_delta_error > 0 and S_boundary > 0",
                                    "score_formula": "boundary_mean_delta_error - max(interior_mean_delta_error, background_mean_delta_error)",
                                    "missing_region_policy": "Samples missing any of the three regions are not ranked",
                                    "sorting": "S_boundary descending; ties by dataset index",
                                    "top_fraction_denominator": "Filtered, rankable candidates separately per mode; round up",
                                    "manual_review": "Inspect exported candidates and choose a structurally clear example",
                                    "candidates": review_summary},
               "dataset_S_B_definition": "Dataset boundary mean delta error - max(dataset interior mean delta error, dataset background mean delta error); apply after region aggregation, not mean of per-image S_boundary; null if any region mean is unavailable",
               "dataset_boundary_scores": dataset_scores,
               "heatmap_range": [-1, 1], "results": summary_rows, "results_macro": macro_rows}
    write_csv(output / "per_sample.csv", rows)
    write_csv(output / "summary.csv", summary_rows)
    write_csv(output / "summary_macro.csv", macro_rows)
    write_csv(output / "summary_S_B.csv", dataset_scores)
    with open(output / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    for row in summary_rows:
        print(f"Pixel-pooled {row['mode']:<8} {row['region']:<20} mean error increase: {row['mean_error_increase']}")
    for row in macro_rows:
        print(f"Image-macro  {row['mode']:<8} {row['region']:<20} mean error increase: {row['mean_error_increase']} (valid images: {row['valid_images']})")
    for row in dataset_scores:
        print(f"Dataset S_B ({row['aggregation']}, {row['mode']}): {row['S_B']}")
    print(f"Saved results to {output.resolve()}")


if __name__ == "__main__":
    main()
