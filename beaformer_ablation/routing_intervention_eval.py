"""Evaluate Normal / Reverse / All-Eb / All-En / Fixed-0.5 on one checkpoint.

    python beaformer_ablation/routing_intervention_eval.py --dataset polypdb_wli \
        --root /path/PolypDB-WLI_224x224 --checkpoint /path/best_model.pth \
        --gpu 0 --output-dir routing_eval_results

All settings come from command-line arguments; config.yaml is not used.
Every encoder/decoder BEA block uses the requested intervention simultaneously.
For x + g*mlp1(x) + (1-g)*mlp2(x), Eb=mlp1 and En=mlp2:
Normal: g; Reverse: 1-g; All-Eb: 1; All-En: 0; Fixed-0.5: 0.5.
Reverse swaps the expert contributions while retaining the router at each block.
Features reaching later blocks may change, so later routers are recomputed.
Weights, residuals, and segmentation head stay intact; no retraining occurs.
Binary masks use logits > 0. Metrics are averaged equally over test images.
ASD here is symmetric ASD (MedPy ASSD), matching the project's metrics_binary.
Distances use the preprocessed pixel grid, without physical voxel spacing.
"""

import argparse
import csv
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from expert_divergence import DATASETS, load_model


MODES = ("Normal", "Reverse", "All-Eb", "All-En", "Fixed-0.5")
METRICS = {"Dice": "dice", "IoU": "iou", "HD95": "hd95", "ASD": "assd"}


class RoutingIntervention:
    def __init__(self, model):
        blocks = [(name, block) for name, block in model.named_modules()
                  if all(hasattr(block, key) for key in ("mlp1", "mlp2", "routing"))]
        if not blocks:
            raise ValueError("No BEA blocks found")
        self.block_paths = [name for name, _ in blocks]
        self.mode, self.calls = "Normal", {}
        self.handles = [block.routing.register_forward_hook(self._hook(name))
                        for name, block in blocks]

    def _hook(self, name):
        def route(module, inputs, g):
            self.calls[name] = self.calls.get(name, 0) + 1
            if self.mode == "Normal":
                return g
            if self.mode == "Reverse":
                return 1 - g
            return torch.full_like(g, {"All-Eb": 1.0, "All-En": 0.0, "Fixed-0.5": 0.5}[self.mode])
        return route

    def predict(self, model, image, mode, probabilities=False):
        if mode not in MODES:
            raise ValueError(f"Unknown intervention: {mode}")
        if model.training:
            raise ValueError("Intervention evaluation requires model.eval()")
        try:
            self.mode, self.calls = mode, {}
            result = model(image)
            if any(self.calls.get(name, 0) != 1 for name in self.block_paths):
                raise RuntimeError("Every BEA router must execute exactly once")
            logits = result[0] if isinstance(result, (tuple, list)) else result
            if logits.ndim != 4 or logits.shape[1] != 1 or not torch.isfinite(logits).all():
                raise ValueError("Expected finite binary logits [B,1,H,W]")
            if probabilities:
                return logits[:, 0].float().sigmoid().cpu()
            return (logits[:, 0] > 0).long().cpu()
        finally:
            self.mode = "Normal"

    def close(self):
        for handle in self.handles:
            handle.remove()


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", default="pvt_v2_b2")
    parser.add_argument("--gpu", type=int, default=0, help="Index among visible CUDA devices")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", required=True)
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

    from metrics import metrics_binary
    name = DATASETS[args.dataset]
    dataset = getattr(importlib.import_module(f"datasets.{name}"), name)(root=args.root, split="test")
    if not len(dataset):
        parser.error("The test set is empty")
    model = load_model(args.checkpoint, num_classes=1, encoder=args.encoder, device=device)
    intervention = RoutingIntervention(model)
    print(f"Intervening on all {len(intervention.block_paths)} BEA blocks", flush=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    scores = {mode: {key: [] for key in METRICS} for mode in MODES}
    try:
        with torch.inference_mode():
            for index in range(len(dataset)):
                sample = dataset[index]
                image = sample["image"].unsqueeze(0).to(device)
                label = sample["label"].cpu()
                if label.ndim != 2 or not ((label == 0) | (label == 1)).all():
                    raise ValueError("Expected binary GT [H,W]")
                for mode in MODES:
                    pred = intervention.predict(model, image, mode)[0]
                    if pred.shape != label.shape:
                        raise ValueError("Prediction/GT shapes differ")
                    values = metrics_binary(pred, label)
                    row = {"dataset_index": index, "source": str(dataset.image_fpaths[index]), "mode": mode}
                    for column, key in METRICS.items():
                        value = float(values[key])
                        if not np.isfinite(value):
                            raise ValueError(f"Non-finite {column} at sample {index}, mode {mode}")
                        scores[mode][column].append(value)
                        row[column] = value
                    rows.append(row)
                print(f"Evaluated {index + 1}/{len(dataset)} (all five modes)", flush=True)
    finally:
        intervention.close()

    summary_rows = [{"mode": mode, **{key: float(np.mean(scores[mode][key])) for key in METRICS}}
                    for mode in MODES]
    summary = {"settings": vars(args), "num_samples": len(dataset),
               "block_paths": intervention.block_paths, "num_blocks": len(intervention.block_paths),
               "routing_weights": {"Normal": "g", "Reverse": "1-g", "All-Eb": 1, "All-En": 0, "Fixed-0.5": 0.5},
               "threshold": "logits > 0", "aggregation": "Equal-weight mean over test images",
               "ASD_definition": "Symmetric average surface distance (MedPy ASSD), in preprocessed pixels",
               "empty_mask_policy": "Both empty: Dice/IoU=1, distances=0; one empty: Dice/IoU=0, distances=224*sqrt(2), matching metrics_binary",
               "results": summary_rows}
    write_csv(output / "per_sample.csv", rows)
    write_csv(output / "summary.csv", summary_rows)
    with open(output / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    print(f"{'Mode':<12} {'Dice':>10} {'IoU':>10} {'HD95':>10} {'ASD':>10}")
    for row in summary_rows:
        print(f"{row['mode']:<12} " + " ".join(f"{row[key]:10.6f}" for key in METRICS))
    print(f"Saved results to {output.resolve()}")


if __name__ == "__main__":
    main()
