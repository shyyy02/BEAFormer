"""Compare raw expert outputs on the paper's six supported test datasets.

Example (run from any directory):
    python beaformer_ablation/expert_divergence.py --dataset polypdb_wli \
        --root /path/PolypDB-WLI_224x224 \
        --unsupervised /path/no_bndsup/best_model.pth \
        --supervised /path/bndsup/best_model.pth --block decoder.stages.2.2 \
        --gpu 0 --output-dir divergence_results --visual-indices 0 10 20

All settings are command-line arguments; config.yaml is not read or modified.

Statistics use D=1-cos(E1(x), E2(x)) computed at native token resolution,
then bilinearly resized to the GT grid (align_corners=False). Boundary is
the project's get_bnd_label; non-boundary is its complement, including
foreground interiors and background. Global S pools pixels over the entire
test set. Per-sample S and its paired macro average are also reported.
No router values enter D; both experts are evaluated in eval mode.
Checkpoint labels describe the supplied training runs, not inferred provenance.
"""

import argparse
import csv
import importlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


DATASETS = {
    "busi": "BUSI",
    "polypdb_wli": "PolypDB_WLI",
    "btmri": "BTMRI",
    "glas": "GlaS",
    "bus_uc": "BUS_UC",
    "kvasirseg": "KvasirSEG",
}


class ExpertCapture:
    """Hooks on whole MLPs capture outputs before drop_path/router weighting."""

    def __init__(self, model, block_path):
        block = model.get_submodule(block_path)
        if not all(hasattr(block, name) for name in ("mlp1", "mlp2", "routing")):
            raise ValueError(f"{block_path} is not a dual-expert BEA block")
        self.outputs = {}
        self.spatial_shape = None
        self.handles = [block.register_forward_pre_hook(self._shape)]
        for name in ("mlp1", "mlp2"):
            self.handles.append(getattr(block, name).register_forward_hook(self._hook(name)))

    def _shape(self, module, inputs):
        x = inputs[0]
        if x.ndim == 4:
            self.spatial_shape = tuple(x.shape[-2:])
        elif x.ndim == 3 and len(inputs) >= 3:
            self.spatial_shape = (int(inputs[1]), int(inputs[2]))
        else:
            raise ValueError("Expected convolutional [B,C,H,W] or PVT [B,N,C] block")

    def _hook(self, name):
        def capture(module, inputs, output):
            if name in self.outputs:
                raise RuntimeError("Selected expert executed more than once per forward")
            self.outputs[name] = output.detach().float()
        return capture

    def divergence(self, model, image):
        self.outputs.clear()
        self.spatial_shape = None
        model(image)  # The segmentation/router return value is deliberately unused.
        if set(self.outputs) != {"mlp1", "mlp2"}:
            raise RuntimeError("Selected experts did not both execute")
        a, b = self.outputs["mlp1"], self.outputs["mlp2"]
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("Expert outputs must have equal shapes and finite values")
        channel_dim = 1 if a.ndim == 4 else 2
        d = 1 - F.cosine_similarity(a, b, dim=channel_dim, eps=1e-8)
        d = d.clamp(0, 2).reshape(a.shape[0], 1, *self.spatial_shape)
        d = F.interpolate(d, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return d[:, 0].cpu().numpy(), self.spatial_shape

    def close(self):
        for handle in self.handles:
            handle.remove()


def load_model(path, num_classes, encoder, device):
    from models.BEAFormer import BEAFormer
    model = BEAFormer(num_classes=num_classes, encoder=encoder)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def region_stats(d, boundary):
    nb, nn = int(boundary.sum()), int((~boundary).sum())
    sb, sn = float(d[boundary].astype(np.float64).sum()), float(d[~boundary].astype(np.float64).sum())
    mb, mn = sb / nb if nb else None, sn / nn if nn else None
    return {"boundary_pixels": nb, "nonboundary_pixels": nn,
            "boundary_sum": sb, "nonboundary_sum": sn,
            "boundary_mean": mb, "nonboundary_mean": mn,
            "S": mb - mn if mb is not None and mn is not None else None}


def plot_comparison(path, image, boundary, unsup, sup, mean=None, std=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rgb = image.cpu().permute(1, 2, 0).numpy()
    if mean is not None and std is not None:
        rgb = np.clip(rgb * np.asarray(std) + np.asarray(mean), 0, 1)
    else:
        rgb = (rgb - rgb.min()) / max(float(np.ptp(rgb)), 1e-8)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[1].imshow(boundary, cmap="gray", vmin=0, vmax=1)
    for ax, d in zip(axes[2:], (unsup, sup)):
        heat = ax.imshow(d, cmap="magma", vmin=0, vmax=2)
    for ax, title in zip(axes, ("Input", "GT Boundary", "Unsupervised Divergence", "Supervised Divergence")):
        ax.set_title(title)
        ax.axis("off")
    fig.colorbar(heat, ax=list(axes[2:]), label="1 - cosine similarity", shrink=0.8)
    fig.savefig(path, dpi=250)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(DATASETS), help="Dataset to evaluate")
    parser.add_argument("--root", required=True, help="Preprocessed dataset root containing test/images and test/labels")
    parser.add_argument("--unsupervised", required=True, help="Full model weights or training checkpoint")
    parser.add_argument("--supervised", required=True, help="Full model weights or training checkpoint")
    parser.add_argument("--encoder", default="pvt_v2_b2")
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--block", default="decoder.stages.2.2", help="Same module path in both models; e.g. encoder.block1.0")
    parser.add_argument("--gpu", type=int, default=3, help="GPU index among visible CUDA devices (default: 0)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--visual-indices", type=int, nargs="+", default=[0, 1, 2, 3], help="Dataset indices, chosen before inspecting results")
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu must be a nonnegative GPU index")
    if args.num_classes < 1:
        parser.error("--num-classes must be positive")
    if Path(args.unsupervised).resolve() == Path(args.supervised).resolve():
        parser.error("Supply two different checkpoint paths")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; this experiment requires a CUDA GPU")
    if args.gpu >= torch.cuda.device_count():
        parser.error(f"--gpu {args.gpu} is unavailable; visible GPU count: {torch.cuda.device_count()}")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    from utils import get_bnd_label
    dataset_name = args.dataset
    root = args.root
    class_name = DATASETS[dataset_name]
    dataset_class = getattr(importlib.import_module(f"datasets.{class_name}"), class_name)
    dataset = dataset_class(root=root, split="test")
    if not len(dataset):
        raise ValueError(f"Empty evaluation set: {root}")
    if any(i < 0 or i >= len(dataset) for i in args.visual_indices):
        parser.error("--visual-indices must be within the evaluation dataset")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    models = [load_model(p, args.num_classes, args.encoder, device)
              for p in (args.unsupervised, args.supervised)]
    captures = [ExpertCapture(m, args.block) for m in models]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    totals = {name: {"boundary_pixels": 0, "nonboundary_pixels": 0,
                     "boundary_sum": 0.0, "nonboundary_sum": 0.0}
              for name in ("unsupervised", "supervised")}
    rows, token_shapes = [], set()
    try:
        with torch.inference_mode():
            for index in range(len(dataset)):
                sample = dataset[index]
                images, labels = sample["image"], sample["label"]
                if images.ndim == 3:
                    images, labels = images.unsqueeze(0), labels.unsqueeze(0)
                if images.ndim != 4 or labels.ndim != 3 or images.shape[0] != labels.shape[0]:
                    raise ValueError("Expected image/label slice tensors or a stack of slices")
                source = str(dataset.image_fpaths[index])
                for slice_index, (image, label) in enumerate(zip(images, labels)):
                    boundary = get_bnd_label(label.cpu()).numpy().astype(bool)
                    maps = []
                    row = {"dataset_index": index, "slice_index": slice_index, "source": source}
                    for name, model, capture in zip(totals, models, captures):
                        d, shape = capture.divergence(model, image.unsqueeze(0).to(device))
                        token_shapes.add(shape)
                        maps.append(d[0])
                        stats = region_stats(d[0], boundary)
                        row.update({f"{name}_{k}": v for k, v in stats.items()})
                        for k in totals[name]:
                            totals[name][k] += stats[k]
                    a, b = row["unsupervised_S"], row["supervised_S"]
                    row["delta_S"] = b - a if a is not None and b is not None else None
                    rows.append(row)
                    if index in args.visual_indices:
                        stem = f"sample_{index:05d}_slice_{slice_index:04d}"
                        np.savez_compressed(output / f"{stem}.npz", boundary=boundary,
                                            unsupervised=maps[0], supervised=maps[1])
                        plot_comparison(output / f"{stem}.png", image, boundary, *maps,
                                        getattr(dataset, "mean", None), getattr(dataset, "std", None))
                print(f"Evaluated {index + 1}/{len(dataset)}", flush=True)
    finally:
        for capture in captures:
            capture.close()

    for name, total in totals.items():
        nb, nn = total["boundary_pixels"], total["nonboundary_pixels"]
        total["boundary_mean"] = total["boundary_sum"] / nb if nb else None
        total["nonboundary_mean"] = total["nonboundary_sum"] / nn if nn else None
        total["S"] = (total["boundary_mean"] - total["nonboundary_mean"]) if nb and nn else None
        scores = [r[f"{name}_S"] for r in rows if r[f"{name}_S"] is not None]
        total["macro_S"] = float(np.mean(scores)) if scores else None
        total["valid_slices_for_macro_S"] = len(scores)
    unsup_s, sup_s = totals["unsupervised"]["S"], totals["supervised"]["S"]
    paired = [r["delta_S"] for r in rows if r["delta_S"] is not None]
    summary = {"settings": vars(args), "resolved_dataset": dataset_name, "resolved_root": root,
               "num_cases": len(dataset), "num_slices": len(rows),
               "native_token_shapes": sorted(token_shapes),
               "definition": "D=1-cos(raw mlp1, raw mlp2); resize D to GT; S=pooled boundary mean - pooled complement mean",
               "heatmap_range": [0, 2], "models": totals,
               "delta_S": sup_s - unsup_s if sup_s is not None and unsup_s is not None else None,
               "paired_macro_delta_S": float(np.mean(paired)) if paired else None,
               "slices_missing_either_region": sum(r["delta_S"] is None for r in rows)}
    with open(output / "per_sample.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(output / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    print(json.dumps(summary["models"], indent=2))
    print(f"S_supervised - S_unsupervised: {summary['delta_S']}")
    print(f"Saved results to {output.resolve()}")


if __name__ == "__main__":
    main()
