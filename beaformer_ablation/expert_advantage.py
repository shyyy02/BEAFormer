"""Expert Advantage Map with all BEA blocks intervened (binary segmentation).

Run with command-line arguments only; config.yaml is neither read nor changed:
    python beaformer_ablation/expert_advantage.py --dataset polypdb_wli \
        --root /path/PolypDB-WLI_224x224 --checkpoint /path/best_model.pth \
        --gpu 0 --output-dir advantage_results

E_b = mlp1 and E_n = mlp2, following x + g*mlp1(x) + (1-g)*mlp2(x).
At ALL encoder and decoder BEA blocks, force g=1 for P_b and g=0 for P_n.
The selected expert has unit weight in every block; residuals stay intact.
Figures show Input | GT Boundary | Expert Advantage Map. Predictions are sigmoid
probabilities, not thresholded masks. A = |P_n-Y| - |P_b-Y|.
Positive/red: E_b is better; negative/blue: E_n is better; zero: tie.
All test images contribute to statistics; visual-indices selects exported maps.
Region statistics partition pixels into boundary (get_bnd_label), interior
(Y=1 outside boundary), and background (Y=0 outside boundary). The existing
nonboundary aggregate is retained as interior + background. Empty regions
have zero counts and null means/fractions.
"""

import argparse
import csv
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from expert_divergence import DATASETS, load_model


class ExpertIntervention:
    """Change all BEA router outputs, without altering model weights."""

    def __init__(self, model):
        blocks = [(name, block) for name, block in model.named_modules()
                  if all(hasattr(block, key) for key in ("mlp1", "mlp2", "routing"))]
        if not blocks:
            raise ValueError("No dual-expert BEA blocks found in model")
        self.block_paths = [name for name, _ in blocks]
        self.mode = "normal"
        self.calls = {}
        self.handles = []
        for name, block in blocks:
            self.handles.append(block.routing.register_forward_hook(self._route(name)))

    def _route(self, name):
        def route(module, inputs, output):
            self.calls[name] = self.calls.get(name, 0) + 1
            if self.mode == "normal":
                return output
            return torch.ones_like(output) if self.mode == "boundary" else torch.zeros_like(output)
        return route

    def _forward(self, model, image, mode):
        self.mode, self.calls = mode, {}
        result = model(image)
        if any(self.calls.get(name, 0) != 1 for name in self.block_paths):
            raise RuntimeError("Every BEA router must execute exactly once per forward")
        logits = result[0] if isinstance(result, (tuple, list)) else result
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError("This experiment expects binary logits [B,1,H,W]")
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite segmentation logits")
        return logits.float().sigmoid()[:, 0]

    def maps(self, model, image, label):
        if model.training:
            raise ValueError("Use model.eval() for deterministic expert interventions")
        try:
            pb = self._forward(model, image, "boundary")
            pn = self._forward(model, image, "nonboundary")
            y = label.to(device=pb.device, dtype=pb.dtype)
            if y.shape != pb.shape or not ((y == 0) | (y == 1)).all():
                raise ValueError("GT must be binary and match prediction shape")
            advantage = (pn - y).abs() - (pb - y).abs()
            return {"P_b": pb.cpu().numpy(), "P_n": pn.cpu().numpy(),
                    "advantage": advantage.cpu().numpy()}
        finally:
            self.mode = "normal"

    def close(self):
        for handle in self.handles:
            handle.remove()


def region_counts(advantage, boundary, label):
    advantage = np.asarray(advantage)
    boundary = np.asarray(boundary, dtype=bool)
    label = np.asarray(label)
    if advantage.shape != boundary.shape or label.shape != boundary.shape:
        raise ValueError("Advantage, boundary and GT must have matching shapes")
    if not np.isin(label, [0, 1]).all():
        raise ValueError("Region statistics require binary GT labels")
    masks = {
        "boundary": boundary,
        "interior": (label == 1) & ~boundary,
        "background": (label == 0) & ~boundary,
        "nonboundary": ~boundary,
    }
    stats = {}
    for name, mask in masks.items():
        values = advantage[mask].astype(np.float64)
        stats[f"{name}_pixels"] = int(values.size)
        stats[f"{name}_advantage_sum"] = float(values.sum())
        stats[f"{name}_Eb_wins"] = int((values > 0).sum())
        stats[f"{name}_En_wins"] = int((values < 0).sum())
        stats[f"{name}_ties"] = int((values == 0).sum())
    return stats


def region_summary(counts):
    result = dict(counts)
    for region in ("boundary", "interior", "background", "nonboundary"):
        n = counts[f"{region}_pixels"]
        result[f"{region}_mean_advantage"] = counts[f"{region}_advantage_sum"] / n if n else None
        for key in ("Eb_wins", "En_wins", "ties"):
            result[f"{region}_{key}_fraction"] = counts[f"{region}_{key}"] / n if n else None
    return result


def plot_maps(path, image, boundary, advantage, dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rgb = image.cpu().permute(1, 2, 0).numpy()
    if hasattr(dataset, "mean") and hasattr(dataset, "std"):
        rgb = np.clip(rgb * np.asarray(dataset.std) + np.asarray(dataset.mean), 0, 1)
    else:
        rgb = (rgb - rgb.min()) / max(float(np.ptp(rgb)), 1e-8)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[1].imshow(boundary, cmap="gray", vmin=0, vmax=1)
    a = axes[2].imshow(advantage, cmap="RdBu_r", vmin=-1, vmax=1)
    for ax, title in zip(axes, ("Input", "GT Boundary", "Expert Advantage Map")):
        ax.set_title(title)
        ax.axis("off")
    fig.colorbar(a, ax=axes[2], shrink=0.8, label="A: blue = En better; red = Eb better")
    fig.savefig(path, dpi=250)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--root", required=True, help="Preprocessed dataset root")
    parser.add_argument("--checkpoint", required=True, help="Full model weights or training checkpoint")
    parser.add_argument("--encoder", default="pvt_v2_b2")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index among visible CUDA devices")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--visual-indices", nargs="+", type=int, default=[0, 1, 2, 3])
    args = parser.parse_args()
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
    if any(i < 0 or i >= len(dataset) for i in args.visual_indices):
        parser.error("--visual-indices must be within the test set")
    model = load_model(args.checkpoint, num_classes=1, encoder=args.encoder, device=device)
    intervention = ExpertIntervention(model)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, totals = [], {}
    try:
        with torch.inference_mode():
            for index in range(len(dataset)):
                sample = dataset[index]
                image, label = sample["image"], sample["label"]
                boundary = get_bnd_label(label.cpu()).numpy().astype(bool)
                maps = intervention.maps(model, image.unsqueeze(0).to(device), label.unsqueeze(0))
                maps = {k: v[0] for k, v in maps.items()}
                counts = region_counts(maps["advantage"], boundary, label.cpu().numpy())
                for k, v in counts.items():
                    totals[k] = totals.get(k, 0) + v
                rows.append({"dataset_index": index, "source": str(dataset.image_fpaths[index]),
                             **region_summary(counts)})
                if index in args.visual_indices:
                    stem = f"sample_{index:05d}"
                    np.savez_compressed(output / f"{stem}.npz", label=label.cpu().numpy(),
                                        boundary=boundary, **maps)
                    plot_maps(output / f"{stem}.png", image, boundary,
                              maps["advantage"], dataset)
                print(f"Evaluated {index + 1}/{len(dataset)}", flush=True)
    finally:
        intervention.close()
    summary = {"settings": vars(args), "num_samples": len(dataset),
               "definition": "A=abs(P_n-Y)-abs(P_b-Y); P_b/P_n are sigmoid full-network predictions",
               "intervention": "ALL BEA blocks: g=1 for Eb=mlp1, g=0 for En=mlp2; residuals retained",
               "block_paths": intervention.block_paths, "num_blocks": len(intervention.block_paths),
               "advantage_range": [-1, 1],
               "region_definitions": {
                   "boundary": "get_bnd_label(Y); includes foreground and background sides of the boundary",
                   "interior": "(Y == 1) and not boundary",
                   "background": "(Y == 0) and not boundary",
                   "nonboundary": "interior + background (retained aggregate)",
               },
               "aggregation": "Pool pixel counts and advantage sums over the full test set",
               "statistics": region_summary(totals)}
    with open(output / "per_sample.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(output / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    print(json.dumps(summary["statistics"], indent=2))
    print(f"Saved results to {output.resolve()}")


if __name__ == "__main__":
    main()
